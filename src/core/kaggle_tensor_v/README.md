# `tensor-v` on a Kaggle box

Clones the `unstable` branch of this repository, builds the library, runs
`tests/test_tensor_v.py`, `tests/test_tensor_v_gate.py` and (when node is
present) `wasm/verify-tensor-v.mjs`, then measures the client/server split with
`tools/bench_tensor_v.py` and solves one real challenge through a
`betterAnusbis` gate in `tensor-v` mode. Everything is copied into
`/kaggle/working/result.json`.

```bash
kaggle kernels push -p src/core/kaggle_tensor_v
kaggle kernels status monktelv31/anubis-tensor-v
kaggle kernels output monktelv31/anubis-tensor-v -p /tmp/tv \
    --file-pattern 'result\.json'
```

The tier is CPU work (the client's integer GEMM and Merkle tree, the server's
96-block sample and proofs), so the kernel is CPU-only on purpose: a CPU kernel
gets four vCPUs on Kaggle, a GPU kernel gets two. The numbers land in
`result.json` under `bench.rows` and are quoted in
[`SPEED.md`](../../../SPEED.md#tensor-v-the-same-gemm-a-sampled-server-check).

`--branch` is fixed to `unstable` in `run.py` because that is the branch that
carries the tier; change it to `main` once it is merged.
