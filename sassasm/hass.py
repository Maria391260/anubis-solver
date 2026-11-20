#!/usr/bin/env python3
"""
HASS - Human-writable SASS
==========================

A readable, editable format for NVIDIA SASS assembly.

Features:
- Named labels (not .L_x_123)
- Preserved comments
- Structured operands
- Optional byte annotations
- Direct binary output via sassasm

Format:
    # Comment lines start with #
    # Labels end with :
    loop_start:
        @P0 BRA `loop_end      ; branch to label
        IADD3 R1, R2, R3, R4    ; comment preserved
        LDC.U8 R5, c[0x3][R6+0x10]

    # Byte offset annotations (optional, for debugging)
    IADD3 R1, R2, R3, R4    // @0x100
"""

import re
import sys
import os
import subprocess
import tempfile
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

# HASS file extension
HASS_EXT = '.hass'
SASS_EXT = '.sass'
BIN_EXT = '.bin'

# Regex patterns for HASS parsing
HASS_LABEL_RE = re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*):\s*(?:#.*)?$')
HASS_INSN_RE = re.compile(r'^\s*([^#;]+?)(?:\s*[#;](.*))?$')
HASS_BRANCH_TARGET_RE = re.compile(r'`([a-zA-Z_][a-zA-Z0-9_]*)')
HASS_BYTE_OFFSET_RE = re.compile(r'//\s*@(0x[0-9a-fA-F]+|\d+)')

# SASS patterns (for sass2hass conversion)
SASS_LABEL_RE = re.compile(r'^\s*(\.L_x_[0-9a-fA-F_]+):')
SASS_INSN_RE = re.compile(r'^\s*/\*([0-9a-f]+)\*/\s+(.+?);')
SASS_BRANCH_TARGET_RE = re.compile(r'`\((\.L_x_[0-9a-fA-F_]+)\)')
SASS_BRANCH_OFFSET_RE = re.compile(r'`(-?\d+)')

# Predicate register names
PRED_REG_NAMES = {i: f'P{i}' for i in range(8)}
PRED_REG_NAMES.update({i+8: f'UP{i}' for i in range(8)})


@dataclass
class HassInstruction:
    """A parsed HASS instruction."""
    label: Optional[str] = None
    mnemonic: str = ''
    operands: List[str] = None
    comment: str = ''
    byte_offset: Optional[int] = None  # Optional byte offset annotation
    
    def __post_init__(self):
        if self.operands is None:
            self.operands = []


@dataclass
class HassProgram:
    """A complete HASS program."""
    instructions: List[HassInstruction] = None
    labels: Dict[str, int] = None  # label -> instruction index
    
    def __post_init__(self):
        if self.instructions is None:
            self.instructions = []
        if self.labels is None:
            self.labels = {}


class HassParser:
    """Parse HASS source into structured representation."""
    
    def __init__(self):
        self.program = HassProgram()
    
    def parse(self, source: str) -> HassProgram:
        """Parse HASS source text."""
        self.program = HassProgram()
        lines = source.split('\n')
        
        for line in lines:
            self._parse_line(line)
        
        return self.program
    
    def _parse_line(self, line: str):
        """Parse a single line."""
        line = line.rstrip('\n\r')
        
        # Skip empty lines
        if not line.strip():
            return
        
        # Check for label
        label_match = HASS_LABEL_RE.match(line)
        if label_match:
            label = label_match.group(1)
            self.program.labels[label] = len(self.program.instructions)
            return
        
        # Parse instruction
        insn_match = HASS_INSN_RE.match(line)
        if not insn_match:
            return
        
        insn_text = insn_match.group(1).strip()
        comment = insn_match.group(2)
        if comment:
            comment = comment.strip()
        
        # Check for byte offset annotation
        byte_offset = None
        if '//' in insn_text:
            parts = insn_text.split('//')
            insn_text = parts[0].strip()
            offset_match = HASS_BYTE_OFFSET_RE.search(parts[1])
            if offset_match:
                offset_str = offset_match.group(1)
                byte_offset = int(offset_str, 0)
        
        # Split mnemonic and operands
        parts = insn_text.split(None, 1)
        mnemonic = parts[0] if parts else ''
        operands_str = parts[1] if len(parts) > 1 else ''
        
        # Parse operands (split by comma, respecting brackets)
        operands = self._parse_operands(operands_str)
        
        # Store instruction
        insn = HassInstruction(
            mnemonic=mnemonic,
            operands=operands,
            comment=comment,
            byte_offset=byte_offset
        )
        self.program.instructions.append(insn)
    
    def _parse_operands(self, operands_str: str) -> List[str]:
        """Parse operand string into list, respecting brackets."""
        operands = []
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
                operands.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        
        if current:
            operands.append(''.join(current).strip())
        
        return [op for op in operands if op]


class HassWriter:
    """Write HASS program to text format."""
    
    @staticmethod
    def write(program: HassProgram, include_offsets: bool = False) -> str:
        """Convert HASS program to text."""
        lines = []
        
        for i, insn in enumerate(program.instructions):
            # Check if this instruction has a label
            for label, idx in program.labels.items():
                if idx == i:
                    lines.append(f'{label}:')
            
            # Build instruction line
            parts = [insn.mnemonic]
            if insn.operands:
                parts.append(', '.join(insn.operands))
            
            line = '    ' + ' '.join(parts)
            
            # Add byte offset annotation if requested
            if include_offsets and insn.byte_offset is not None:
                line += f'  // @0x{insn.byte_offset:x}'
            elif include_offsets:
                line += f'  // @0x{i * 16:x}'
            
            # Add comment
            if insn.comment:
                line += f'  ; {insn.comment}'
            
            lines.append(line)
        
        return '\n'.join(lines)


class HassAssembler:
    """Assemble HASS directly to binary using sassasm."""
    
    def __init__(self, arch: str = 'turing', sassasm_path: str = './sassasm'):
        self.arch = arch
        self.sassasm_path = sassasm_path
    
    def assemble(self, program: HassProgram, output_bin: str) -> bool:
        """Assemble HASS program to binary."""
        # Convert to SASS with resolved branches
        sass_lines = self._to_sass(program)
        
        # Write temp SASS file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sass', delete=False) as f:
            f.write('\n'.join(sass_lines) + '\n')
            temp_sass = f.name
        
        try:
            # Assemble using sassasm
            cmd = [self.sassasm_path, '-a', self.arch, '-f', temp_sass, '-o', output_bin]
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=os.path.dirname(self.sassasm_path) or '.')
            if result.returncode != 0:
                print(f"Assembly failed: {result.stderr}")
                return False
            
            # Generate listing
            listing_file = output_bin + '.txt'
            cmd_text = [self.sassasm_path, '-a', self.arch, '-f', temp_sass]
            result_text = subprocess.run(cmd_text, capture_output=True, text=True,
                                       cwd=os.path.dirname(self.sassasm_path) or '.')
            if result_text.returncode == 0:
                with open(listing_file, 'w') as f:
                    f.write(result_text.stdout)
                print(f"  Listing: {listing_file}")
            
            return True
        finally:
            os.unlink(temp_sass)
    
    def _to_sass(self, program: HassProgram) -> List[str]:
        """Convert HASS to SASS format with branch resolution."""
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
                    # Resolve branch targets
                    op = self._resolve_branch_target(op, i, label_to_idx)
                    operands.append(op)
                parts.append(', '.join(operands))
            
            line = ' '.join(parts) + ' ;'
            sass_lines.append(line)
        
        return sass_lines
    
    def _resolve_branch_target(self, operand: str, current_idx: int, 
                                labels: Dict[str, int]) -> str:
        """Resolve label references in branch operands."""
        match = HASS_BRANCH_TARGET_RE.search(operand)
        if match:
            label = match.group(1)
            if label in labels:
                target_idx = labels[label]
                byte_offset = (target_idx - current_idx) * 16
                # Replace label with backtick-offset format
                operand = operand.replace(f'`{label}', f'`{byte_offset}')
        return operand


def load_hass_file(filepath: str) -> HassProgram:
    """Load HASS program from file."""
    with open(filepath, 'r') as f:
        source = f.read()
    parser = HassParser()
    return parser.parse(source)


def save_hass_file(program: HassProgram, filepath: str, include_offsets: bool = False):
    """Save HASS program to file."""
    content = HassWriter.write(program, include_offsets)
    with open(filepath, 'w') as f:
        f.write(content)


if __name__ == '__main__':
    # Test parsing
    test_hass = """
# Test HASS program
main:
    @P0 BRA `exit          ; conditional branch
    IADD3 R1, R2, R3, R4    ; add three registers
    LDC.U8 R5, c[0x3][R6+0x10]  ; load constant
    SHF.R.U32 R7, R8, 0x10, R9  ; shift funnel
    
loop:
    IMAD R10, R11, R12, R13     ; multiply-add
    @!P1 BRA `loop       ; branch back
    
exit:
    EXIT                      ; terminate
"""
    
    parser = HassParser()
    program = parser.parse(test_hass)
    
    print("Parsed program:")
    print(f"  Instructions: {len(program.instructions)}")
    print(f"  Labels: {program.labels}")
    
    print("\nWritten HASS:")
    print(HassWriter.write(program, include_offsets=True))