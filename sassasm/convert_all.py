#!/usr/bin/env python3
"""
Batch convert all SASS files to HASS and verify round-trip.
"""

import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(__file__))

SASS_DIR = "/media/miku6/kubuntu_2404/home/miku/DOWN/pyani/anubis-solver/src/nvidia/sass"
SASS_FILES = [
    "nvidiaT4_sha256.sass",
    "nvidiaT4_sha256_opt.sass",
    "nvidiaT4_hashx.sass",
    "nvidiaT4_hashx_opt.sass",
    "nvidiaT4_argon2id.sass",
    "nvidiaT4_argon2id_opt.sass",
]

def run_cmd(cmd, cwd=None):
    """Run command and return (success, output)."""
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return result.returncode == 0, result.stdout, result.stderr

def convert_sass_to_hass(sass_file, hass_file):
    """Convert SASS to HASS."""
    cmd = [sys.executable, 'sass2hass.py', sass_file, '-o', hass_file, '--offsets']
    return run_cmd(cmd)

def convert_hass_to_bin(hass_file, bin_file):
    """Assemble HASS to binary."""
    cmd = [sys.executable, 'hass2sass.py', hass_file, '-b', bin_file]
    return run_cmd(cmd)

def compare_bins(bin1, bin2):
    """Compare two binary files."""
    with open(bin1, 'rb') as f1, open(bin2, 'rb') as f2:
        data1 = f1.read()
        data2 = f2.read()
    return data1 == data2

def main():
    print("=" * 60)
    print("Batch SASS -> HASS -> Binary Conversion & Verification")
    print("=" * 60)
    
    all_passed = True
    
    for sass_file in SASS_FILES:
        sass_path = os.path.join(SASS_DIR, sass_file)
        if not os.path.exists(sass_path):
            print(f"SKIP: {sass_file} not found")
            continue
        
        base = os.path.splitext(sass_file)[0]
        hass_file = f"/tmp/{base}.hass"
        bin_file = f"/tmp/{base}_rt.bin"  # round-trip binary
        orig_bin = f"/tmp/{base}.bin"
        
        print(f"\n{'='*60}")
        print(f"Processing: {sass_file}")
        print(f"{'='*60}")
        
        # Step 1: SASS -> HASS
        print("  Step 1: SASS -> HASS...", end=' ')
        success, stdout, stderr = convert_sass_to_hass(sass_path, hass_file)
        if not success:
            print(f"FAILED\n{stderr}")
            all_passed = False
            continue
        print("OK")
        
        # Step 2: HASS -> Binary
        print("  Step 2: HASS -> Binary...", end=' ')
        success, stdout, stderr = convert_hass_to_bin(hass_file, bin_file)
        if not success:
            print(f"FAILED\n{stderr}")
            all_passed = False
            continue
        print("OK")
        
        # Step 3: Compare with original binary
        print("  Step 3: Verifying round-trip...", end=' ')
        if os.path.exists(orig_bin):
            if compare_bins(orig_bin, bin_file):
                print("MATCH")
            else:
                print("MISMATCH!")
                all_passed = False
        else:
            print(f"SKIP (original binary not found: {orig_bin})")
        
        # Show stats
        hass_size = os.path.getsize(hass_file)
        bin_size = os.path.getsize(bin_file)
        print(f"  HASS size: {hass_size:,} bytes")
        print(f"  Binary size: {bin_size:,} bytes ({bin_size//16} instructions)")
    
    print(f"\n{'='*60}")
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()