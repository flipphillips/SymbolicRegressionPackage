# SymbolicRegressionPackage (Fork)

This fork adds GPU acceleration and parallel execution support for the PyTorch training scripts.

## Changes
- **GPU Acceleration:** Enabled CUDA support in `tree_prototype_torch_v16_final.py` with automated device detection.
- **Parallel Workers:** Added support for running multiple seeds in parallel using `torch.multiprocessing` (spawn method).
- **VRAM Optimization:** Designed to utilize high-memory GPUs (like the NVIDIA A6000) by batching multiple training runs concurrently.
- **Environment Isolation:** Updated headless scripts to clear `PYTHONPATH`, preventing conflicts with system-level packages (e.g., RenderMan).

## Usage
The training scripts will automatically use the GPU if available.

### Parallel Execution
To run multiple seeds concurrently, use the `--workers` flag:
```bash
python3 tree_prototype_torch_v16_final.py --workers 8 ...
```
For a 48GB A6000, you can typically scale this to 16-32 workers for small tree depths to significantly speed up sweeps.
