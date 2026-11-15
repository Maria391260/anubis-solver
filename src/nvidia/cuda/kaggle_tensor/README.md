# The `tensor` tier on Kaggle's 2x Tesla T4

Clones this repository, builds with `CUDA_ARCH=sm_75` (the T4's compute
capability, which is what enables the `__dp4a` path), runs
`tests/test_tensor_cuda.py`, then benchmarks the int8 GEMM on one T4 and on
both, plus one end-to-end challenge through the gate.

```bash
kaggle kernels push -p src/nvidia/cuda/kaggle_tensor
kaggle kernels status monktelv31/anubis-t4-tensor
kaggle kernels output monktelv31/anubis-t4-tensor -p /tmp/t4 \
    --file-pattern 'result\.json'
```

The numbers land in `result.json` (`one_t4`, `two_t4`) and are quoted in
[`SPEED.md`](../../../SPEED.md#cuda-nvidia-gpus).
