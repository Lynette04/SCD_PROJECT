"""
check_quantisation.py

Shows the int8 quantisation formulas working on REAL weights from one of your saved models.
Read-only: it loads the model, copies one layer's weights into numpy, and prints results.
It does not change or save any model.

Run from your project folder (where the .pt files are):
    python check_quantisation.py sickle_cell_edge_model.pt
You can also try:
    python check_quantisation.py sickle_cell_model.pt
    python check_quantisation.py sickle_cell_effnet_lite0.pt
To look at a different layer, add its position in the list printed at the top:
    python check_quantisation.py sickle_cell_edge_model.pt 12
"""
import sys

import numpy as np


# ---------- the formulas ----------
def quantise_asymmetric(r):
    """Per-tensor int8: r ~ S * (q - Z), with q clipped to [-128, 127]."""
    r_min, r_max = float(r.min()), float(r.max())
    S = (r_max - r_min) / 255.0
    Z = int(round(-128 - r_min / S))
    q = np.clip(np.round(r / S + Z), -128, 127).astype(np.int8)
    return q, S, Z


def dequantise_asymmetric(q, S, Z):
    return S * (q.astype(np.float32) - Z)


def quantise_per_channel_symmetric(r):
    """Per-output-channel int8 with zero point 0 (the style TensorFlow Lite uses for weights)."""
    flat = r.reshape(r.shape[0], -1)                   # first axis = output channels in PyTorch (Keras keeps them last)
    scales = np.abs(flat).max(axis=1, keepdims=True) / 127.0
    scales[scales == 0] = 1.0
    q = np.clip(np.round(flat / scales), -127, 127).astype(np.int8)
    recovered = (q.astype(np.float32) * scales).reshape(r.shape)
    return q.reshape(r.shape), scales, recovered


def report(r, label):
    r = r.astype(np.float32)
    q, S, Z = quantise_asymmetric(r)
    back = dequantise_asymmetric(q, S, Z)
    err = np.abs(r - back)

    print(f"\n{label}")
    print(f"  number of weights: {r.size:,}   (float32: {r.size * 4:,} bytes, int8: {r.size:,} bytes)")
    print(f"  smallest = {r.min():.6f}   largest = {r.max():.6f}")
    print(f"  scale S = {S:.8f}   zero point Z = {Z}")
    print(f"  largest possible rounding error is half a step: S/2 = {S / 2:.8f}")

    print("\n  first 8 weights:  original -> stored int8 -> recovered   (error)")
    for o, qi, b in list(zip(r.flatten()[:8], q.flatten()[:8], back.flatten()[:8])):
        print(f"    {o: .6f} -> {int(qi):>4} -> {b: .6f}   ({abs(o - b):.6f})")

    print(f"\n  all weights, per-tensor asymmetric:  mean error = {err.mean():.8f}   biggest error = {err.max():.8f}")

    if r.ndim in (2, 4):
        _, _, rec_pc = quantise_per_channel_symmetric(r)
        err_pc = np.abs(r - rec_pc)
        print(f"  all weights, per-channel symmetric:  mean error = {err_pc.mean():.8f}   biggest error = {err_pc.max():.8f}")
        print("  (per-channel keeps a separate scale for each output channel, which usually lowers the error)")


# ---------- load a real model ----------
def main():
    from pytorch_training import load_model

    path = sys.argv[1] if len(sys.argv) > 1 else "sickle_cell_edge_model.pt"
    model = load_model(path)

    #Convolution and dense kernels are the weights with 4 or 2 dimensions.
    #Batch normalisation weights have 1 dimension, so they are left out.
    kernels = []
    for name, w in model.state_dict().items():
        if name.endswith("weight") and w.ndim in (2, 4):
            kernels.append((name, w))
    if not kernels:
        raise SystemExit("No convolution or dense kernels found in this model.")

    print(f"Model: {path}")
    print(f"Weight tensors found (position: name, shape):")
    for i, (name, w) in enumerate(kernels):
        print(f"  {i:>3}: {name}  {tuple(w.shape)}")

    pick = int(sys.argv[2]) if len(sys.argv) > 2 else len(kernels) // 2
    name, w = kernels[pick]
    report(w.detach().cpu().numpy(), f"Layer {pick}: {name}  shape {tuple(w.shape)}")


if __name__ == "__main__":
    main()