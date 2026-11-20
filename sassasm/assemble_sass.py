#!/usr/bin/env python3
"""Extract supported instructions from SASS files and assemble them with label resolution."""

import re
import subprocess
import sys
import os
import tempfile

SASSASM = "./sassasm"
ARCH = "turing"

# Patterns to skip (not yet supported)
SKIP_PATTERNS = [
    r'^\s*\.',           # directives (.section, .global, etc.)
    r'^\s*_',            # labels (_Z...)
]

# Branch instructions that use labels
BRANCH_INSNS = {'BRA', 'BSSY', 'BRX', 'BAR.SYNC', 'BRA.'}

LABEL_RE = re.compile(r'^\s*(\.L_x_[0-9a-fA-F_]+):')

# Instruction line pattern: /*XXXX*/  INSN ;
INSN_RE = re.compile(r'^\s*/\*([0-9a-f]+)\*/\s+(.+?);')

# Branch target pattern: `(.L_x_123)
BRANCH_TARGET_RE = re.compile(r'`\((\.L_x_[0-9a-fA-F_]+)\)')

def is_supported(line):
    """Check if instruction is supported."""
    line = line.strip()
    if not line or line.startswith('//'):
        return False
    for pat in SKIP_PATTERNS:
        if re.search(pat, line):
            return False
    return True

def extract_instructions(sass_file):
    """Extract supported instructions from SASS file with label resolution."""
    instructions = []
    labels = {}  # label_name -> instruction_index
    branch_refs = []  # (insn_index, branch_target_label)
    
    with open(sass_file, 'r') as f:
        lines = f.readlines()
    
    # First pass: collect labels and instructions
    insn_index = 0
    for line in lines:
        line = line.rstrip('\n')
        
        # Check for label definition
        label_match = LABEL_RE.match(line)
        if label_match:
            label_name = label_match.group(1)
            labels[label_name] = insn_index
            continue
        
        # Match instruction lines
        m = INSN_RE.match(line)
        if m:
            insn = m.group(2).strip()
            # Strip trailing comment annotations like (*"BRANCH_TARGETS ..."*)
            if '(*"' in insn:
                insn = insn.split('(*"')[0].strip()
            if is_supported(insn):
                # Check for branch target
                branch_match = BRANCH_TARGET_RE.search(insn)
                if branch_match:
                    target_label = branch_match.group(1)
                    branch_refs.append((insn_index, target_label))
                
                instructions.append((insn, insn_index))
                insn_index += 1
    
    return instructions, labels, branch_refs

def resolve_branches(insns_data, labels, branch_refs):
    """Resolve branch targets to byte offsets."""
    resolved = []
    for insn_text, insn_index in insns_data:
        resolved_text = insn_text
        for ref_idx, label in branch_refs:
            if ref_idx == insn_index and label in labels:
                target_insn_index = labels[label]
                # Each instruction is 16 bytes
                byte_offset = (target_insn_index - insn_index) * 16
                # Replace the label reference with backtick + byte offset
                # Format: `(.L_x_123) -> `3248
                old_target = f'`({label})'
                new_target = f'`{byte_offset}'
                resolved_text = resolved_text.replace(old_target, new_target)
        resolved.append((resolved_text, insn_index))
    return resolved

def assemble_instructions(insns, output_bin):
    """Assemble instructions to binary."""
    if not insns:
        return False
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sass', delete=False) as f:
        for insn in insns:
            f.write(insn + '\n')
        temp_file = f.name
    
    try:
        # Assemble
        cmd = [SASSASM, '-a', ARCH, '-f', temp_file, '-o', output_bin]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(SASSASM))
        if result.returncode != 0:
            print(f"Assembly failed: {result.stderr}")
            return False
        
        # Also output text listing
        cmd_text = [SASSASM, '-a', ARCH, '-f', temp_file]
        result_text = subprocess.run(cmd_text, capture_output=True, text=True, cwd=os.path.dirname(SASSASM))
        if result_text.returncode == 0:
            listing_file = output_bin + '.txt'
            with open(listing_file, 'w') as f:
                f.write(result_text.stdout)
            print(f"  Listing: {listing_file}")
        
        return True
    finally:
        os.unlink(temp_file)

def main():
    sass_dir = "/media/miku6/kubuntu_2404/home/miku/DOWN/pyani/anubis-solver/src/nvidia/sass"
    sass_files = [
        "nvidiaT4_sha256.sass",
        "nvidiaT4_sha256_opt.sass",
        "nvidiaT4_hashx.sass",
        "nvidiaT4_hashx_opt.sass",
        "nvidiaT4_argon2id.sass",
        "nvidiaT4_argon2id_opt.sass",
    ]
    
    for sass_file in sass_files:
        path = os.path.join(sass_dir, sass_file)
        if not os.path.exists(path):
            print(f"Skipping {sass_file} (not found)")
            continue
        
        print(f"\nProcessing {sass_file}...")
        insns_data, labels, branch_refs = extract_instructions(path)
        print(f"  Found {len(insns_data)} supported instructions, {len(labels)} labels")
        
        if insns_data:
            # Resolve branch targets
            resolved_insns = resolve_branches(insns_data, labels, branch_refs)
            insns = [i[0] for i in resolved_insns]
            
            output_bin = f"/tmp/{sass_file.replace('.sass', '.bin')}"
            if assemble_instructions(insns, output_bin):
                size = os.path.getsize(output_bin)
                print(f"  Assembled: {output_bin} ({size} bytes, {size//16} instructions)")
            else:
                print(f"  Assembly failed")
        else:
            print(f"  No supported instructions found")

if __name__ == '__main__':
    main()