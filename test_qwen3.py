#!/usr/bin/env python3
"""
Test script for Qwen3 Python bindings.

This script tests both chat and embedding models.
Model files are stored in ./model subfolder.
"""

import os
import time

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

def test_chat_model():
    """Test Qwen3 chat model."""
    print("=" * 60)
    print("Testing Qwen3 Chat Model")
    print("=" * 60)

    from qwen3_infer import Qwen3Model

    # Load model (downloads if not present)
    print("\n[1] Loading model...")
    start = time.time()
    model = Qwen3Model(
        model_name="Qwen/Qwen3-0.6B",
        model_dir=MODEL_DIR,
        model_type="chat"
    )
    load_time = time.time() - start
    print(f"    Model loaded in {load_time:.2f}s")

    # Test generation
    print("\n[2] Testing generation...")
    prompts = [
        "Introduce yourself。",
        "What is 2 + 2?",
    ]

    for prompt in prompts:
        print(f"\n    Prompt: {prompt}")
        start = time.time()
        response = model.generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.7,
            no_think=True
        )
        gen_time = time.time() - start
        print(f"    Response: {response}")
        print(f"    Time: {gen_time:.2f}s")

    print("\n[3] Chat model test completed!")
    return True


def test_embedding_model():
    """Test Qwen3 embedding model."""
    print("\n" + "=" * 60)
    print("Testing Qwen3 Embedding Model")
    print("=" * 60)

    from qwen3_infer import Qwen3Model

    # Load model
    print("\n[1] Loading model...")
    start = time.time()
    model = Qwen3Model(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_dir=MODEL_DIR,
        model_type="embedding"
    )
    load_time = time.time() - start
    print(f"    Model loaded in {load_time:.2f}s")

    # Test embedding
    print("\n[2] Testing embedding...")
    texts = [
        "what is your name",
        "where is shanghai",
        "apple is red",
        "what is the difference between banana and organge？",
    ]

    for text in texts:
        print(f"\n    Text: {text}")
        start = time.time()
        embedding = model.embed(text)
        embed_time = time.time() - start
        print(f"    Dimension: {len(embedding)}")
        print(f"    First 5 values: {embedding[:5]}")
        print(f"    Time: {embed_time:.2f}s")

    print("\n[3] Embedding model test completed!")
    return True


def main():
    """Run all tests."""
    print("\nQwen3 Python Bindings Test")
    print(f"Model directory: {MODEL_DIR}\n")

    try:
        # Test chat model
       ## test_chat_model()

        # Test embedding model
        test_embedding_model()

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    except ImportError as e:
        print(f"\nError: Could not import qwen3_infer module.")
        print(f"Make sure you have built and installed the package:")
        print(f"  pip install maturin")
        print(f"  maturin develop --release")
        print(f"\nOriginal error: {e}")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
