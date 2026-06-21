#!/usr/bin/env python3
"""Generate small .npy fixtures for the c4tts npy IO test.

These are written by NumPy and read back by the C++ npy loader, so they prove
cross-language compatibility of the format we use to exchange golden vectors.

Usage: gen_test_fixtures.py <out_dir>
"""
import os
import sys

import numpy as np


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "fixtures"
    os.makedirs(out_dir, exist_ok=True)

    # 2-D float32 matrix: arange(12).reshape(3, 4)
    mat = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(os.path.join(out_dir, "mat_f32.npy"), mat)

    # 1-D float64 (loader must down-cast to float32)
    vec = np.array([1.5, -2.25, 3.125], dtype=np.float64)
    np.save(os.path.join(out_dir, "vec_f64.npy"), vec)

    # 1-D int64 token ids (loader must down-cast to int32)
    ids = np.array([10, 20, 30, 8193], dtype=np.int64)
    np.save(os.path.join(out_dir, "ids_i64.npy"), ids)

    print(f"wrote fixtures to {out_dir}")


if __name__ == "__main__":
    main()
