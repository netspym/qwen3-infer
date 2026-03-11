#!/usr/bin/env python3
"""
Memory usage measurement script for Qwen3 embedding model.
Tracks RAM usage before, during, and after model loading and inference.
"""

import os
import time
import psutil
import gc

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")


def get_memory_info():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size (actual RAM)
        'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
    }


def print_memory(label, mem_info):
    """Print memory usage with label."""
    print(f"{label:30s} RSS: {mem_info['rss_mb']:8.2f} MB | VMS: {mem_info['vms_mb']:8.2f} MB")


def main():
    print("=" * 70)
    print("Qwen3 Embedding Model - Memory Usage Measurement")
    print("=" * 70)

    # System info
    print(f"\nSystem RAM: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.2f} GB")
    print(f"Available RAM: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.2f} GB")

    # Baseline
    gc.collect()
    time.sleep(0.5)
    baseline = get_memory_info()
    print_memory("\n[Baseline] Before import", baseline)

    # Import
    from qwen3_infer import Qwen3Model
    gc.collect()
    time.sleep(0.5)
    after_import = get_memory_info()
    print_memory("[Import] After import", after_import)
    print(f"  → Import overhead: {after_import['rss_mb'] - baseline['rss_mb']:.2f} MB")

    # Load model
    print(f"\n[Loading] Model: Qwen/Qwen3-Embedding-0.6B")
    print(f"[Loading] Directory: {MODEL_DIR}")
    start = time.time()

    model = Qwen3Model(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_dir=MODEL_DIR,
        model_type="embedding"
    )

    load_time = time.time() - start
    gc.collect()
    time.sleep(0.5)
    after_load = get_memory_info()
    print_memory("\n[Loaded] After model load", after_load)
    print(f"  → Model load time: {load_time:.2f}s")
    print(f"  → Model memory: {after_load['rss_mb'] - after_import['rss_mb']:.2f} MB")

    # Run inference (single embedding)
    print("\n[Inference] Running single embedding...")
    test_text = "葆婴益生菌的成份含量"
    start = time.time()
    embedding = model.embed(test_text)
    infer_time = time.time() - start

    gc.collect()
    time.sleep(0.5)
    after_single = get_memory_info()
    print_memory("[Inference] After single embed", after_single)
    print(f"  → Inference time: {infer_time:.4f}s")
    print(f"  → Embedding dim: {len(embedding)}")
    print(f"  → Memory delta: {after_single['rss_mb'] - after_load['rss_mb']:.2f} MB")

    # Run batch inference (like your test - 24 embeddings)
    print("\n[Batch] Running 24 embeddings...")
    texts = [
        "葆婴益生菌的成份含量",
        "usana cellsentials的成份含量",
    ] * 12

    start = time.time()
    embeddings = [model.embed(text) for text in texts]
    batch_time = time.time() - start

    gc.collect()
    time.sleep(0.5)
    after_batch = get_memory_info()
    print_memory("[Batch] After 24 embeddings", after_batch)
    print(f"  → Total time: {batch_time:.2f}s")
    print(f"  → Avg time per embedding: {batch_time/len(texts):.4f}s")
    print(f"  → Peak memory delta: {after_batch['rss_mb'] - after_load['rss_mb']:.2f} MB")

    # Peak usage
    peak = max(after_load['rss_mb'], after_single['rss_mb'], after_batch['rss_mb'])
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline memory:          {baseline['rss_mb']:8.2f} MB")
    print(f"Model memory:             {after_load['rss_mb'] - baseline['rss_mb']:8.2f} MB")
    print(f"Peak total memory:        {peak:8.2f} MB")
    print(f"Inference overhead:       {after_batch['rss_mb'] - after_load['rss_mb']:8.2f} MB")
    print("=" * 70)

    print("\n💡 To reduce memory by 50%, use F16 precision:")
    print("   model = Qwen3Model(..., dtype='f16')")
    print("   (requires code modification as discussed)")


if __name__ == "__main__":
    main()
