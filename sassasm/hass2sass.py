#!/usr/bin/env python3
"""
hass2sass - Convert Human-writable SASS (HASS) to SASS or binary

Usage:
    python3 hass2sass.py input.hass -o output.sass     # Convert to SASS
    python3 hass2sass.py input.hass -b output.bin      # Assemble to binary
    python3 hass2sass.py input.hass                    # Print SASS to stdout
"""

import re
import sys
import os
import argparse
import subprocess
import tempfile
from typing import List, Dict

# Import HASS module
sys.path.insert(0, os.path.dirname(__file__))
from hass import (HassProgram, HassInstruction, HassParser, HassWriter, 
                  HassAssembler, load_hass_file, save_hass_file,
                  HASS_BRANCH_TARGET_RE)


def hass2sass(program: HassProgram, output_file: str = None) -> List[str]:
    """Convert HASS program to SASS format."""
    # First pass: collect all labels and their instruction indices
    label_to_idx = dict(program.labels)
    
    # Second pass: generate SASS lines
    sass_lines = []
    
    for i, insn in enumerate(program.instructions):
        # Build SASS instruction
        parts = [insn.mnemonic]
        if insn.operands:
            operands = []
            for op in insn.operands:
                # Resolve branch targets: `label -> `(.L_x_...)
                # For SASS output, we need to convert HASS labels back to .L_x_ format
                # Since we don't have the original .L_x_ names, we'll use the hass label
                # with a prefix to make it valid SASS
                op = resolve_branch_for_sass(op, i, label_to_idx)
                operands.append(op)
            parts.append(', '.join(operands))
        
        line = ' '.join(parts) + ' ;'
        sass_lines.append(line)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(sass_lines) + '\n')
        print(f"Converted to SASS: {output_file}")
    
    return sass_lines


def resolve_branch_for_sass(operand: str, current_idx: int, 
                            labels: Dict[str, int]) -> str:
    """Resolve HASS label references to SASS branch offset format."""
    def replace_branch_target(match):
        label = match.group(1)
        if label in labels:
            target_idx = labels[label]
            byte_offset = (target_idx - current_idx) * 16
            return f'`{byte_offset}'
        return f'`{label}'  # Keep as-is if not found
    
    return HASS_BRANCH_TARGET_RE.sub(replace_branch_target, operand)


def hass2bin(program: HassProgram, output_bin: str, 
             arch: str = 'turing', sassasm_path: str = './sassasm') -> bool:
    """Assemble HASS program directly to binary."""
    assembler = HassAssembler(arch=arch, sassasm_path=sassasm_path)
    return assembler.assemble(program, output_bin)


def main():
    parser = argparse.ArgumentParser(description='Convert HASS to SASS or binary')
    parser.add_argument('input', help='Input HASS file')
    parser.add_argument('-o', '--output', help='Output SASS file')
    parser.add_argument('-b', '--binary', help='Output binary file (assemble)')
    parser.add_argument('-a', '--arch', default='turing', 
                       help='Target architecture (default: turing)')
    parser.add_argument('--sassasm', default='./sassasm',
                       help='Path to sassasm binary')
    parser.add_argument('--stdout', action='store_true',
                       help='Write SASS to stdout')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Load HASS program
    print(f"Loading {args.input}...")
    program = load_hass_file(args.input)
    print(f"  Instructions: {len(program.instructions)}")
    print(f"  Labels: {len(program.labels)}")
    
    if args.binary:
        # Assemble to binary
        print(f"Assembling to {args.binary}...")
        success = hass2bin(program, args.binary, args.arch, args.sassasm)
        if success:
            size = os.path.getsize(args.binary)
            print(f"  Success: {size} bytes ({size//16} instructions)")
        else:
            print("  Assembly failed")
            sys.exit(1)
    else:
        # Convert to SASS
        if args.stdout:
            sass_lines = hass2sass(program)
            print('\n'.join(sass_lines))
        elif args.output:
            hass2sass(program, args.output)
        else:
            # Default: print to stdout
            sass_lines = hass2sass(program)
            print('\n'.join(sass_lines))


if __name__ == '__main__':
    main()