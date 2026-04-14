# SymbolicRegressionPackage (Fork)

This fork adds GPU acceleration support for the PyTorch training scripts.

## Changes
- Enabled CUDA support in `tree_prototype_torch_v16_final.py`.
- Automated device detection (GPU/CPU).
- Optimized tensor placement for NVIDIA A6000 and similar GPUs.

## Usage
The training scripts will automatically use the GPU if available.
