#!/usr/bin/env python3
"""
sass2hass - Convert NVIDIA SASS to Human-writable SASS (HASS)

Usage:
    python3 sass2hass.py input.sass -o output.hass
    python3 sass2hass.py input.sass > output.hass
"""

import re
import sys
import os
import argparse
from typing import Dict, List, Tuple

# Import HASS module
sys.path.insert(0, os.path.dirname(__file__))
from hass import HassProgram, HassInstruction, HassWriter, PRED_REG_NAMES

# SASS regex patterns
SASS_LABEL_RE = re.compile(r'^\s*(\.L_x_[0-9a-fA-F_]+):')
SASS_INSN_RE = re.compile(r'^\s*/\*([0-9a-f]+)\*/\s+(.+?);')
SASS_BRANCH_TARGET_RE = re.compile(r'`\((\.L_x_[0-9a-fA-F_]+)\)')
SASS_BRANCH_OFFSET_RE = re.compile(r'`(-?\d+)')
SASS_COMMENT_RE = re.compile(r'//.*$')

# Label mapping: .L_x_123 -> meaningful names
def generate_label_name(sass_label: str, label_map: Dict[str, str], label_counts: Dict[str, int]) -> str:
    """Convert .L_x_123 to readable label."""
    if sass_label in label_map:
        return label_map[sass_label]
    
    # Extract numeric part for naming
    match = re.search(r'\.L_x_([0-9a-fA-F_]+)', sass_label)
    if match:
        num_part = match.group(1)
        # Try to make meaningful name
        base_name = f'label_{num_part}'
    else:
        base_name = 'label_unknown'
    
    # Handle duplicates
    if base_name in label_counts:
        label_counts[base_name] += 1
        unique_name = f'{base_name}_{label_counts[base_name]}'
    else:
        label_counts[base_name] = 0
        unique_name = base_name
    
    label_map[sass_label] = unique_name
    return unique_name


def parse_sass_file(sass_file: str) -> Tuple[List[Tuple[str, int]], Dict[str, int], List[Tuple[int, str]]]:
    """
    Parse SASS file and extract instructions, labels, and branch references.
    Returns: (instructions, labels, branch_refs)
    instructions: list of (instruction_text, instruction_index)
    labels: dict of sass_label -> instruction_index
    branch_refs: list of (instruction_index, target_sass_label)
    """
    with open(sass_file, 'r') as f:
        lines = f.readlines()
    
    instructions = []
    labels = {}
    branch_refs = []
    insn_index = 0
    
    for line in lines:
        line = line.rstrip('\n')
        
        # Check for label definition
        label_match = SASS_LABEL_RE.match(line)
        if label_match:
            label = label_match.group(1)
            labels[label] = insn_index
            continue
        
        # Match instruction lines
        m = SASS_INSN_RE.match(line)
        if m:
            insn = m.group(2).strip()
            # Strip trailing comment annotations like (*"BRANCH_TARGETS ..."*)
            if '(*"' in insn:
                insn = insn.split('(*"')[0].strip()
            
            # Check for branch target
            branch_match = SASS_BRANCH_TARGET_RE.search(insn)
            if branch_match:
                target_label = branch_match.group(1)
                branch_refs.append((insn_index, target_label))
            
            instructions.append((insn, insn_index))
            insn_index += 1
    
    return instructions, labels, branch_refs


def convert_operand(operand: str, label_map: Dict[str, str]) -> str:
    """Convert a SASS operand to HASS format."""
    # Handle branch targets: `(.L_x_123) -> `label_name
    def replace_branch_target(match):
        sass_label = match.group(1)
        return f'`{label_map.get(sass_label, sass_label)}'
    
    # Handle branch offsets: `3248 -> `label_name (if we can resolve)
    def replace_branch_offset(match):
        offset = match.group(1)
        # Could try to resolve offset to label, but keep as-is for now
        return f'`{offset}'
    
    operand = SASS_BRANCH_TARGET_RE.sub(replace_branch_target, operand)
    operand = SASS_BRANCH_OFFSET_RE.sub(replace_branch_offset, operand)
    
    return operand


def convert_instruction(insn_text: str, label_map: Dict[str, str]) -> HassInstruction:
    """Convert a SASS instruction to HASS format."""
    # Split mnemonic and operands
    parts = insn_text.split(None, 1)
    mnemonic = parts[0] if parts else ''
    operands_str = parts[1] if len(parts) > 1 else ''
    
    # Parse operands (split by comma, respecting brackets)
    operands = []
    if operands_str:
        current = []
        bracket_depth = 0
        for ch in operands_str:
            if ch == '[':
                bracket_depth += 1
                current.append(ch)
            elif ch == ']':
                bracket_depth -= 1
                current.append(ch)
            elif ch == ',' and bracket_depth == 0:
                operand = ''.join(current).strip()
                if operand:
                    operands.append(convert_operand(operand, label_map))
                current = []
            else:
                current.append(ch)
        
        if current:
            operand = ''.join(current).strip()
            if operand:
                operands.append(convert_operand(operand, label_map))
    
    return HassInstruction(mnemonic=mnemonic, operands=operands)


def sass2hass(sass_file: str, output_file: str = None, include_offsets: bool = False) -> HassProgram:
    """Convert SASS file to HASS program."""
    # Parse SASS
    instructions, sass_labels, branch_refs = parse_sass_file(sass_file)
    
    # Generate label mapping
    label_map = {}
    label_counts = {}
    for sass_label, idx in sass_labels.items():
        hass_label = generate_label_name(sass_label, label_map, label_counts)
        # Verify the mapping
        assert label_map[sass_label] == hass_label
    
    # Convert instructions
    hass_instructions = []
    for insn_text, insn_idx in instructions:
        insn = convert_instruction(insn_text, label_map)
        
        # Add byte offset if requested
        if include_offsets:
            insn.byte_offset = insn_idx * 16
        
        hass_instructions.append(insn)
    
    # Build program
    program = HassProgram(
        instructions=hass_instructions,
        labels={v: k for k, v in sass_labels.items()}  # hass_label -> index
    )
    # Fix label mapping (hass_label -> index)
    program.labels = {label_map[k]: v for k, v in sass_labels.items()}
    
    # Save if output file specified
    if output_file:
        save_hass_file(program, output_file, include_offsets)
        print(f"Converted {sass_file} -> {output_file}")
        print(f"  Instructions: {len(program.instructions)}")
        print(f"  Labels: {len(program.labels)}")
    
    return program


def save_hass_file(program: HassProgram, filepath: str, include_offsets: bool = False):
    """Save HASS program to file."""
    content = HassWriter.write(program, include_offsets)
    with open(filepath, 'w') as f:
        f.write(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert SASS to HASS')
    parser.add_argument('input', help='Input SASS file')
    parser.add_argument('-o', '--output', help='Output HASS file')
    parser.add_argument('--offsets', action='store_true', 
                       help='Include byte offset annotations')
    parser.add_argument('--stdout', action='store_true',
                       help='Write to stdout instead of file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if args.stdout:
        output_file = None
    elif args.output:
        output_file = args.output
    else:
        # Default: same name with .hass extension
        base = os.path.splitext(args.input)[0]
        output_file = base + '.hass'
    
    program = sass2hass(args.input, output_file, args.offsets)
    
    if args.stdout:
        print(HassWriter.write(program, args.offsets))