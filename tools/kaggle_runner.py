#!/usr/bin/env python3
"""
Kaggle Kernel Runner for anubis-solver Benchmarks
==================================================

This script is designed to be copy-pasted into a Kaggle kernel notebook
or run as a standalone script in a Kaggle environment.

It will:
1. Install/clone the anubis-solver repository
2. Build the native library (make)
3. Run comprehensive benchmarks
4. Output results as CSV for Kaggle dataset creation

Usage in Kaggle Notebook:
```python
!git clone https://github.com/TecharoHQ/anubis-solver.git
%cd anubis-solver
!make -j$(nproc)
!python3 tools/kaggle_bench.py --reps 3 --threads 4 --output /kaggle/working/results.csv
```

Or upload this file as a dataset and run:
```python
%run kaggle_runner.py --reps 3 --threads 4
```
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, cwd=None, capture=True):
    """Run command and return (success, stdout, stderr)."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True, cwd=cwd)
    if capture:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    return result.returncode == 0, result.stdout, result.stderr

def main():
    parser = argparse.ArgumentParser(description="Kaggle Runner for anubis-solver benchmarks")
    parser.add_argument("--repo", default="https://github.com/TecharoHQ/anubis-solver.git",
                       help="Repository URL")
    parser.add_argument("--branch", default="main", help="Git branch")
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per benchmark")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU benchmarks")
    parser.add_argument("--output", default="/kaggle/working/kaggle_bench.csv",
                       help="Output CSV path")
    parser.add_argument("--skip-build", action="store_true", help="Skip build step")
    args = parser.parse_args()
    
    # Detect if running in Kaggle
    in_kaggle = os.path.exists("/kaggle/working")
    if in_kaggle:
        print("Running in Kaggle environment")
        work_dir = Path("/kaggle/working")
    else:
        work_dir = Path.cwd()
    
    repo_dir = work_dir / "anubis-solver"
    
    # Step 1: Clone repository
    if not args.skip_build:
        if not repo_dir.exists():
            print(f"Cloning {args.repo}...")
            success, _, _ = run_cmd(f"git clone --branch {args.branch} {args.repo} {repo_dir}")
            if not success:
                sys.exit(1)
        
        # Step 2: Build
        print("Building anubis-solver...")
        success, _, _ = run_cmd("make -j$(nproc)", cwd=repo_dir)
        if not success:
            print("Build failed!", file=sys.stderr)
            sys.exit(1)
    
    # Step 3: Run benchmarks
    print("\nRunning benchmarks...")
    bench_cmd = (
        f"python3 tools/kaggle_bench.py "
        f"--reps {args.reps} "
        f"--threads {args.threads} "
        f"--output {args.output}"
    )
    if args.cpu_only:
        bench_cmd += " --cpu-only"
    
    success, _, _ = run_cmd(bench_cmd, cwd=repo_dir)
    if not success:
        print("Benchmarks failed!", file=sys.stderr)
        sys.exit(1)
    
    # Step 4: Create Kaggle dataset metadata
    metadata = {
        "id": "kaggle-user/anubis-solver-benchmarks",
        "licenses": [{"name": "MIT"}],
        "title": "anubis-solver Benchmark Results",
        "subtitle": f"Benchmarks from {repo_dir.name} with {args.threads} threads, {args.reps} reps",
        "description": "Auto-generated benchmark results from anubis-solver",
        "tags": ["benchmark", "anubis", "proof-of-work", "cpu", "gpu", "avx"],
        "files": [
            {"name": Path(args.output).name, "description": "Benchmark results CSV"}
        ]
    }
    
    import json
    metadata_path = Path(args.output).with_name("dataset-metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nCreated dataset metadata: {metadata_path}")
    
    print(f"\n✅ Benchmarks complete! Results saved to {args.output}")
    if in_kaggle:
        print("You can now create a Kaggle dataset from /kaggle/working/")
        print("  Dataset > Create New Dataset > Select /kaggle/working/")

if __name__ == "__main__":
    main()