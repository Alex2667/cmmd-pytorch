"""
Is exaclty as them main.py but handles directory of folders with datasets and saves the results in root. 
Handles also single folders and saves a csv.
Works with several GPUs by distributing the folders across them.
Example usage:
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=0 python batch_cmmd.py "" /export/data/abespalo/eval/coconut_base_stepdependency3 --ref_embed_file=/export/data/abespalo/datasets/unsplash-research-dataset-lite-latest/unsplash_images_all/ref_embeddings.npy --batch_size=32 
"""

# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The main entry point for the CMMD calculation."""

from absl import app
from absl import flags
import distance
import embedding
import io_util
import numpy as np
import os
import csv
from pathlib import Path
import torch
import torch.multiprocessing as mp
import math

_BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "Batch size for embedding generation.")
_MAX_COUNT = flags.DEFINE_integer("max_count", -1, "Maximum number of images to read from each directory.")
_REF_EMBED_FILE = flags.DEFINE_string(
    "ref_embed_file", None, "Path to the pre-computed embedding file for the reference images."
)

# Valid extensions to identify image folders
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

def is_image_folder(path):
    """Checks if a directory contains image files."""
    path = Path(path)
    if not path.is_dir():
        return False
    for file in path.iterdir():
        if file.suffix.lower() in IMAGE_EXTENSIONS:
            return True
    return False

def compute_cmmd(ref_dir, eval_dir, ref_embed_file=None, batch_size=32, max_count=-1, embedding_model=None):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
      ref_dir: Path to the directory containing reference images.
      eval_dir: Path to the directory containing images to be evaluated.
      ref_embed_file: Path to the pre-computed embedding file for the reference images.
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.

    Returns:
      The CMMD value between the image sets.
    """
    if ref_dir and ref_embed_file:
        raise ValueError("`ref_dir` and `ref_embed_file` both cannot be set at the same time.")
    if embedding_model is None:
        embedding_model = embedding.ClipEmbeddingModel()

    # Handle Reference Embeddings
    if ref_embed_file is not None:
        ref_embs = np.load(ref_embed_file).astype("float32")
    else:
        ref_embs = io_util.compute_embeddings_for_dir(ref_dir, embedding_model, batch_size, max_count).astype(
            "float32"
        )
        if ref_dir: 
            save_path = os.path.join(ref_dir, "ref_embeddings.npy")
            print(f"Saving reference embeddings to {save_path}...")
            np.save(save_path, ref_embs)
        
    eval_embs = io_util.compute_embeddings_for_dir(eval_dir, embedding_model, batch_size, max_count).astype("float32")
    val = distance.mmd(ref_embs, eval_embs)
    return val.numpy()

def worker_process(gpu_id, folder_chunk, ref_dir, ref_embed_file, batch_size, max_count, return_queue):
    """
    Worker function that runs on a specific GPU.
    It loads the model ONCE and processes all assigned folders.
    """
    try:
        # Set the specific GPU for this process
        torch.cuda.set_device(gpu_id)
        print(f"[GPU {gpu_id}] Initializing model...")
        
        # Load model on this specific GPU
        model = embedding.ClipEmbeddingModel()
        
        # Process the chunk of folders
        results = []
        for folder_path in folder_chunk:
            print(f"[GPU {gpu_id}] Processing: {folder_path.name}")
            try:
                val = compute_cmmd(
                    ref_dir=ref_dir,
                    eval_dir=str(folder_path),
                    ref_embed_file=ref_embed_file,
                    batch_size=batch_size,
                    max_count=max_count,
                    embedding_model=model
                )
                results.append({"Folder": folder_path.name, "CMMD": val})
            except Exception as e:
                print(f"[GPU {gpu_id}] Error on {folder_path.name}: {e}")
                results.append({"Folder": folder_path.name, "CMMD": "Error"})
        
        # Send results back to main process
        return_queue.put(results)
        
    except Exception as e:
        print(f"[GPU {gpu_id}] CRITICAL WORKER ERROR: {e}")
        return_queue.put([])


def main(argv):
    if len(argv) != 3:
        raise app.UsageError("Too few/too many command-line arguments.")
    
    _, ref_dir, eval_root_str = argv
    
    # 1. Setup paths
    eval_root = Path(eval_root_str)
    if not eval_root.exists():
        print(f"Error: Directory {eval_root} does not exist.")
        return

    # 2. Identify all folders to process (Root + Subdirectories)
    tasks = []
    
    # Check if the root folder itself has images
    if is_image_folder(eval_root):
        tasks.append(eval_root)
        
    # Check immediate subdirectories
    for item in eval_root.iterdir():
        if item.is_dir() and is_image_folder(item):
            # Avoid adding root again if it was already added
            if item != eval_root:
                tasks.append(item)

    if not tasks:
        print(f"No image folders found in {eval_root}.")
        return

    # 2. Setup GPU Parallelism
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found! Falling back to CPU (single process).")
        num_gpus = 1 # Fallback logic could be added, but assuming GPU usage here.

    print(f"Found {len(tasks)} folders. Distributing across {num_gpus} GPU(s).")

    # Split tasks into chunks
    chunk_size = math.ceil(len(tasks) / num_gpus)
    chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]

    # 3. Spawn Processes
    mp.set_start_method('spawn', force=True) # Critical for CUDA
    queue = mp.Queue()
    processes = []

    for i in range(len(chunks)):
        # If we have more chunks than GPUs (unlikely with this logic), wrap around
        gpu_id = i % num_gpus
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, chunks[i], ref_dir, _REF_EMBED_FILE.value, _BATCH_SIZE.value, _MAX_COUNT.value, queue)
        )
        p.start()
        processes.append(p)

    # 4. Collect Results
    final_results = []
    for _ in processes:
        # Block until we get a result from a worker
        worker_results = queue.get()
        final_results.extend(worker_results)

    # Wait for all to finish
    for p in processes:
        p.join()

    # 5. Save Aggregated Results
    csv_path = eval_root / "cmmd_results.csv"
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['Folder', 'CMMD']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        # Sort results by folder name for cleanliness
        final_results.sort(key=lambda x: x['Folder'])
        writer.writerows(final_results)

    print(f"\nAll processing complete. Results saved to: {csv_path}")

if __name__ == "__main__":
    app.run(main)