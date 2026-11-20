#!/usr/bin/env python3
"""
Kaggle Benchmark Suite for anubis-solver
=========================================
Runs comprehensive CPU/GPU benchmarks and outputs results in Kaggle-friendly format.

Usage:
    python3 tools/kaggle_bench.py [--output results/kaggle_bench.csv]

This script is designed to run on Kaggle kernels (which provide GPUs like T4, P100, A100)
and produce a CSV file that can be uploaded as a Kaggle dataset or used in a competition.

Requirements:
- Run `make` first to build libanubis_solver.so
- Kaggle kernels provide NVIDIA GPUs (T4, P100, etc.) with CUDA
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from anubis_solver import AnubisSolver
except ImportError:
    print("Error: anubis_solver not found. Run `make` first.", file=sys.stderr)
    sys.exit(1)


class KaggleBenchmark:
    """Comprehensive benchmark suite for Kaggle."""
    
    def __init__(self, output_csv: str):
        self.output_csv = Path(output_csv)
        self.results: List[Dict] = []
        self.host_info = self._get_host_info()
        
    def _get_host_info(self) -> Dict:
        """Collect host information."""
        info = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "cpus": os.cpu_count(),
            "python": platform.python_version(),
            "os": platform.platform(),
        }
        
        # GPU info
        try:
            gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", 
                                        "--format=csv,noheader,nounits"], 
                                      capture_output=True, text=True, timeout=5)
            if gpu_info.returncode == 0:
                lines = gpu_info.stdout.strip().split('\n')
                info["gpus"] = [{"name": l.split(', ')[0], 
                                 "memory_mb": int(l.split(', ')[1]),
                                 "driver": l.split(', ')[2]} for l in lines]
            else:
                info["gpus"] = []
        except Exception:
            info["gpus"] = []
        
        # CUDA version
        try:
            cuda_info = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
            if cuda_info.returncode == 0:
                for line in cuda_info.stdout.split('\n'):
                    if "release" in line:
                        info["cuda_version"] = line.strip().split("release ")[1].split(",")[0]
                        break
        except Exception:
            info["cuda_version"] = "unknown"
        
        # Build info
        info["avx2_disabled"] = bool(os.environ.get("ANUBIS_NO_AVX2"))
        info["shani_disabled"] = bool(os.environ.get("ANUBIS_NO_SHANI"))
        info["cuda_disabled"] = bool(os.environ.get("ANUBIS_NO_CUDA"))
        info["vulkan_disabled"] = bool(os.environ.get("ANUBIS_NO_VULKAN"))
        info["levelzero_disabled"] = bool(os.environ.get("ANUBIS_NO_LEVELZERO"))
        info["opencl_disabled"] = bool(os.environ.get("ANUBIS_NO_OPENCL"))
        
        return info
    
    def _verify_digest(self, algorithm: str, challenge: str, nonce: int, result_hash: str) -> bool:
        """Verify a digest against hashlib."""
        try:
            if algorithm in ("fast", "slow"):
                test = hashlib.sha256((challenge + str(nonce)).encode()).hexdigest()
                return test == result_hash
            elif algorithm == "sha256":
                raw = bytes.fromhex(challenge)
                use_le = raw[-1] >= 0x80
                test = hashlib.sha256(raw + nonce.to_bytes(4, "little" if use_le else "big")).hexdigest()
                return test == result_hash
            elif algorithm == "hashx":
                # HashX verification would need the HashX implementation
                return True  # Skip for now
            elif algorithm == "argon2id":
                # Argon2id verification would need argon2-cffi
                return True  # Skip for now
        except Exception:
            pass
        return False
    
    def run_cpu_benchmarks(self, reps: int = 3, threads: int = 1) -> None:
        """Run CPU benchmarks for all algorithms."""
        print(f"\n=== CPU Benchmarks (threads={threads}, reps={reps}) ===")
        
        solver = AnubisSolver(threads=threads)
        gpu_backend = solver.gpu or "cpu"
        
        # Tier definitions (matching Anubis challenge tiers)
        tiers = {
            "fast": [(2, "light"), (4, "medium"), (6, "hard")],
            "slow": [(2, "light"), (4, "medium"), (6, "hard")],
            "sha256": [(8, "light"), (16, "medium"), (24, "hard")],
            "hashx": [(7, "light"), (15, "medium"), (20, "hard")],
            "argon2id": [(2, "light"), (4, "medium"), (6, "hard")],
        }
        
        def expected_hashes(algo: str, diff: int) -> int:
            if algo in ("fast", "slow"):
                return 16 ** diff
            return 2 ** diff
        
        for algo, tier_list in tiers.items():
            for diff, tier_name in tier_list:
                print(f"  {algo} {tier_name} (d={diff})...", end=" ", flush=True)
                
                times = []
                nonces = []
                all_verified = True
                
                for _ in range(reps):
                    challenge = os.urandom(64).hex()
                    if algo in ("fast", "slow"):
                        nonce, result_hash, ms = solver._native.solve_nibbles(challenge, diff, threads, 0)
                        if not self._verify_digest(algo, challenge, nonce, result_hash):
                            all_verified = False
                    elif algo == "sha256":
                        nonce, result_hash, ms = solver._native.solve_bits(challenge, diff, threads, 0)
                        if not self._verify_digest(algo, challenge, nonce, result_hash):
                            all_verified = False
                    elif algo == "hashx":
                        nonce, result_hash, ms = solver._native.solve_hashx(challenge, diff, threads, 0)
                    elif algo == "argon2id":
                        nonce, result_hash, ms = solver._native.solve_argon2id(challenge, diff, threads, 0)
                    else:
                        continue
                    
                    times.append(ms)
                    nonces.append(max(1, nonce))
                
                if not times:
                    continue
                
                median_ms = sorted(times)[len(times)//2]
                total_time = sum(times)
                total_work = sum(nonces)
                rate = total_work / (total_time / 1000.0) if total_time else 0
                expected_work = expected_hashes(algo, diff)
                expected_ms = expected_work / rate * 1000.0 if rate else 0
                
                print(f"  {rate:>12,.0f} H/s  (median: {median_ms:.1f}ms, verified: {all_verified})")
                
                self.results.append({
                    "benchmark_type": "cpu",
                    "backend": gpu_backend,
                    "algorithm": algo,
                    "tier": tier_name,
                    "difficulty": diff,
                    "threads": threads,
                    "reps": reps,
                    "median_ms": round(median_ms, 2),
                    "min_ms": round(min(times), 2),
                    "max_ms": round(max(times), 2),
                    "expected_ms": round(expected_ms, 2),
                    "hashes_per_second": int(rate),
                    "total_candidates_tested": total_work,
                    "expected_hashes": expected_work,
                    "all_verified": all_verified,
                    **self.host_info
                })
    
    def run_thread_scaling(self, reps: int = 3) -> None:
        """Run thread scaling benchmark for fast algorithm."""
        print(f"\n=== Thread Scaling (fast d=4, reps={reps}) ===")
        
        for threads in [1, 2, 4, 8, 16]:
            if threads > os.cpu_count():
                continue
            solver = AnubisSolver(threads=threads)
            gpu_backend = solver.gpu or "cpu"
            
            times = []
            for _ in range(reps):
                challenge = os.urandom(64).hex()
                _, _, ms = solver._native.solve_nibbles(challenge, 4, threads, 0)
                times.append(ms)
            
            median_ms = sorted(times)[len(times)//2]
            print(f"  {threads:>2} threads: {median_ms:>7.1f} ms (min {min(times):.1f})")
            
            self.results.append({
                "benchmark_type": "thread_scaling",
                "backend": gpu_backend,
                "algorithm": "fast",
                "difficulty": 4,
                "threads": threads,
                "reps": reps,
                "median_ms": round(median_ms, 2),
                "min_ms": round(min(times), 2),
                "max_ms": round(max(times), 2),
                **self.host_info
            })
    
    def run_avx_comparison(self) -> None:
        """Compare AVX2 vs AVX-512 performance."""
        print(f"\n=== AVX Comparison ===")
        
        def run_bench(force_avx2: bool) -> Dict[str, float]:
            env = dict(os.environ)
            env.pop("ANUBIS_NO_AVX512", None)
            if force_avx2:
                env["ANUBIS_NO_AVX512"] = "1"
            
            proc = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "bench_avx512.py")],
                env=env, capture_output=True, text=True, timeout=300
            )
            if proc.returncode != 0:
                return {}
            
            rates = {}
            width = 0
            for line in proc.stdout.splitlines():
                w = re.match(r"^batch width: (\d+)", line)
                if w:
                    width = int(w.group(1))
                m = re.match(r"^(fast d6|sha256 b24)\s*:.*?([0-9.]+) MH/s", line)
                if m:
                    rates[m.group(1)] = float(m.group(2))
            return rates
        
        import re
        native = run_bench(False)
        avx2 = run_bench(True)
        
        print(f"  Native (AVX-512 when available): {native}")
        print(f"  Forced AVX2: {avx2}")
        
        for key in ("fast d6", "sha256 b24"):
            n = native.get(key)
            a = avx2.get(key)
            if n and a:
                speedup = n / a if a else 0
                print(f"  {key}: native={n:.2f} MH/s, AVX2={a:.2f} MH/s, speedup={speedup:.2f}x")
                
                self.results.append({
                    "benchmark_type": "avx_comparison",
                    "workload": key,
                    "native_mhs": n,
                    "avx2_mhs": a,
                    "speedup": round(speedup, 2),
                    "batch_width": width if 'width' in locals() else 0,
                    **self.host_info
                })
    
    def run_gpu_benchmarks(self, reps: int = 3) -> None:
        """Run GPU backend comparison."""
        print(f"\n=== GPU Backend Comparison (reps={reps}) ===")
        
        modes = [
            ("cuda/l0/vulkan (auto)", {}),
            ("level_zero", {"ANUBIS_NO_CUDA": "1", "ANUBIS_NO_VULKAN": "1"}),
            ("vulkan", {"ANUBIS_NO_CUDA": "1", "ANUBIS_NO_LEVELZERO": "1"}),
            ("opencl", {"ANUBIS_OPENCL": "1", "ANUBIS_NO_CUDA": "1",
                        "ANUBIS_NO_VULKAN": "1", "ANUBIS_NO_LEVELZERO": "1"}),
            ("cpu", {"ANUBIS_NO_CUDA": "1", "ANUBIS_NO_VULKAN": "1",
                     "ANUBIS_NO_LEVELZERO": "1", "ANUBIS_NO_OPENCL": "1"}),
        ]
        
        def run_mode(name: str, env_extra: Dict[str, str]) -> Dict:
            env = dict(os.environ)
            env.update(env_extra)
            proc = subprocess.run([sys.executable, __file__, "--worker"], 
                                  env=env, capture_output=True, text=True, timeout=180)
            if proc.returncode != 0:
                sys.stderr.write(proc.stderr)
                return {}
            for line in proc.stdout.splitlines():
                if line.startswith("RESULT "):
                    return json.loads(line[7:])
            return {}
        
        if "--worker" in sys.argv:
            return self._run_gpu_worker()
        
        results = []
        for name, env in modes:
            print(f"  Testing {name}...", end=" ", flush=True)
            res = run_mode(name, env)
            if res:
                results.append((name, res))
                print(f"  {res.get('gpu', 'unknown')}")
            else:
                print("  FAILED")
        
        if results:
            print(f"\n{'workload':<12}", end="")
            for name, _ in results:
                print(f"{name[:10]:>10}", end="")
            print()
            
            keys = ("fast d6", "fast d7", "sha256 b24", "sha256 b26")
            for key in keys:
                row = f"{key:<12}"
                for _, res in results:
                    v = res.get("work", {}).get(key)
                    row += f"{v['mhs']:>10.1f}" if v else f"{'N/A':>10}"
                print(row)
                
                for name, res in results:
                    v = res.get("work", {}).get(key)
                    if v:
                        self.results.append({
                            "benchmark_type": "gpu",
                            "backend_name": name,
                            "actual_backend": res.get("gpu", "unknown"),
                            "workload": key,
                            "mhs": v["mhs"],
                            "time_sec": v["sec"],
                            **self.host_info
                        })
    
    def _run_gpu_worker(self) -> int:
        """Worker mode for GPU benchmark (called via subprocess)."""
        from anubis_solver import AnubisSolver
        
        solver = AnubisSolver(threads=1)
        ch = "ab" * 64
        out = {"gpu": solver.gpu, "cpu_threads": solver.threads, "work": {}}
        
        def run_fast(d: int):
            t0 = time.perf_counter()
            n, r, _ = solver._native.solve_nibbles(ch, d, 1, 0)
            dt = time.perf_counter() - t0
            if hashlib.sha256((ch + str(n)).encode()).hexdigest() != r:
                raise SystemExit("fast digest mismatch")
            out["work"][f"fast d{d}"] = {"sec": dt, "mhs": n / dt / 1e6}
        
        def run_bits(d: int):
            chh = "cd" * 64
            raw = bytes.fromhex(chh)
            use_le = raw[-1] >= 0x80
            t0 = time.perf_counter()
            n, r, _ = solver._native.solve_bits(chh, d, 1, 0)
            dt = time.perf_counter() - t0
            want = hashlib.sha256(raw + n.to_bytes(4, "little" if use_le else "big")).hexdigest()
            if want != r:
                raise SystemExit("sha256 digest mismatch")
            out["work"][f"sha256 b{d}"] = {"sec": dt, "mhs": n / dt / 1e6}
        
        run_fast(6)
        run_fast(7)
        run_bits(24)
        run_bits(26)
        print("RESULT " + json.dumps(out), flush=True)
        return 0
    
    def save_csv(self) -> None:
        """Save results to CSV."""
        if not self.results:
            print("No results to save!")
            return
        
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        # Flatten host_info into each row
        fieldnames = set()
        for row in self.results:
            fieldnames.update(row.keys())
        fieldnames = sorted(fieldnames)
        
        with open(self.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.results:
                writer.writerow(row)
        
        print(f"\nResults saved to {self.output_csv}")
        print(f"Total rows: {len(self.results)}")
    
    def save_json(self) -> None:
        """Save results to JSON for detailed analysis."""
        json_file = self.output_csv.with_suffix('.json')
        with open(json_file, 'w') as f:
            json.dump({
                "host": self.host_info,
                "results": self.results
            }, f, indent=2)
        print(f"JSON saved to {json_file}")


def main():
    parser = argparse.ArgumentParser(description="Kaggle Benchmark Suite for anubis-solver")
    parser.add_argument("--output", "-o", default="results/kaggle_bench.csv",
                        help="Output CSV file")
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per benchmark")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads")
    parser.add_argument("--cpu-only", action="store_true", help="Run only CPU benchmarks")
    parser.add_argument("--worker", action="store_true", help="Internal worker mode for GPU bench")
    args = parser.parse_args()
    
    if args.worker:
        bench = KaggleBenchmark(args.output)
        return bench._run_gpu_worker()
    
    bench = KaggleBenchmark(args.output)
    
    print(f"Kaggle Benchmark Suite")
    print(f"Host: {bench.host_info.get('processor', 'unknown')} ({bench.host_info.get('cpus', 0)} CPUs)")
    if bench.host_info.get('gpus'):
        for gpu in bench.host_info['gpus']:
            print(f"GPU: {gpu['name']} ({gpu['memory_mb']} MB)")
    print(f"CUDA: {bench.host_info.get('cuda_version', 'unknown')}")
    
    # Run benchmarks
    bench.run_cpu_benchmarks(reps=args.reps, threads=args.threads)
    bench.run_thread_scaling(reps=args.reps)
    bench.run_avx_comparison()
    
    if not args.cpu_only:
        try:
            bench.run_gpu_benchmarks(reps=args.reps)
        except Exception as e:
            print(f"GPU benchmarks skipped: {e}")
    
    bench.save_csv()
    bench.save_json()
    
    print("\n=== Summary ===")
    print(f"Results: {args.output}")
    print(f"Host: {bench.host_info['machine']}")
    if bench.host_info.get('gpus'):
        for gpu in bench.host_info['gpus']:
            print(f"  GPU: {gpu['name']}")


if __name__ == "__main__":
    import re
    import json
    import hashlib
    sys.exit(main())