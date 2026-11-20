# HASS - Human-writable SASS

**HASS** (Human-writable SASS) is a readable, editable format for NVIDIA SASS assembly that converts losslessly to/from binary via the `sassasm` assembler.

## Quick Start

```bash
# Convert SASS to HASS (human-readable)
python3 sass2hass.py kernel.sass -o kernel.hass --offsets

# Convert HASS back to SASS
python3 hass2sass.py kernel.hass -o kernel.sass

# Assemble HASS directly to binary
python3 hass2sass.py kernel.hass -b kernel.bin

# Round-trip verification
python3 convert_all.py
```

## HASS Format Specification

### Comments
```hass
# Full-line comment
INSN  R1, R2, R3  ; Inline comment
```

### Labels
```hass
label_name:
    INSN  R1, R2, R3
    BRA  `target_label   ; Branch to label
```

Labels must start with a letter or underscore, followed by alphanumerics/underscores.

### Instructions
```hass
MNEMONIC  operand1, operand2, operand3
```

Operands are comma-separated. Brackets `[ ]` are preserved for memory operands.

### Branch Targets
```hass
BRA  `target_label      ; Branch to label (resolved to byte offset)
BSSY  B0, `sync_point   ; Barrier sync
```

Branch targets use backtick syntax: `` `label ``. These are automatically resolved to PC-relative byte offsets during assembly.

### Byte Offset Annotations (Optional)
```hass
IADD3  R1, R2, R3, R4    // @0x100
```

The `// @0xADDRESS` annotation is optional and ignored during assembly. Useful for debugging/diffing.

## Predication
```hass
@P0     INSN  ...           ; Execute if P0 true
@!P0    INSN  ...           ; Execute if P0 false
@UPT    INSN  ...           ; Uniform predicate
@!UP1   INSN  ...           ; Negated uniform predicate
```

## Register Types
| Type | Syntax | Range |
|------|--------|-------|
| GPR | `R0`-`R255` | General purpose |
| Predicate | `P0`-`P7` | Standard predicates |
| Uniform Predicate | `UP0`-`UP7` | Uniform predicates |
| Uniform Register | `UR0`-`UR7` | Uniform registers |
| Special | `RZ`, `SP`, `SR_*`, `B0`-`B15` | Special registers |

## Memory Operands
```hass
[R1]                ; Register indirect
[R1+0x10]           ; Register + offset
[UR4+0x400]         ; Uniform register base
[0x400]             ; Immediate address
[R2.64+UR6]         ; 64-bit address + UR
```

## Constant Memory
```hass
c[0x0][0x28]        ; Bank 0, offset 0x28
c[0x3][R12+0x100]   ; Bank 3, register + offset
```

## Reuse Flag
```hass
SHF.R.U32.HI  R2, RZ, 0x8, R0.reuse
```

## Tools

### sass2hass.py - SASS → HASS
```bash
python3 sass2hass.py input.sass -o output.hass [--offsets]
```
Options:
- `--offsets` - Include byte offset annotations
- `--stdout` - Write to stdout

### hass2sass.py - HASS → SASS / Binary
```bash
python3 hass2sass.py input.hass -o output.sass     # To SASS
python3 hass2sass.py input.hass -b output.bin      # To binary
python3 hass2sass.py input.hass                    # To stdout (SASS)
```
Options:
- `-a, --arch` - Target architecture (default: turing)
- `--sassasm` - Path to sassasm binary

### convert_all.py - Batch Conversion
```bash
python3 convert_all.py
```
Converts all 6 Anubis kernels and verifies round-trip.

## Example: Writing a Kernel from Scratch

```hass
# My SHA-256 Kernel
# ==================

# Constants
THREADS_PER_BLOCK: 256

# Entry point
kernel_main:
    # Get thread ID
    S2R     R0, SR_TID.X
    S2R     R1, SR_CTAID.X
    
    # Bounds check
    IMAD    R2, R1, c[0x0][0x0], R0
    ISETP.GE.U32.AND  P0, PT, R2, c[0x0][0x100], PT
    @P0     EXIT

    # Load data
    LDC.U8  R3, c[0x2][R0+0x20]
    LDC.U8  R4, c[0x2][R0+0x21]

    # Compute
    IMAD    R5, R3, R4, RZ
    IADD3   R6, R5, R5, RZ

    # Store result
    STL     [R7], R6

    # Loop
loop_start:
    IADD3   R0, R0, 1, RZ
    ISETP.LT.U32.AND  P0, PT, R0, 64, PT
    @!P0    BRA  `loop_end
    BRA     `loop_start

loop_end:
    EXIT
```

## Round-trip Guarantee

HASS ↔ SASS ↔ Binary conversion is **lossless** for all supported instructions:

```bash
# This produces identical binary:
sass2hass.py kernel.sass -o kernel.hass
hass2sass.py kernel.hass -b kernel.bin

# Verified for all 6 Anubis kernels (sha256, hashx, argon2id × 2)
```

## Supported Architectures

Currently: **Turing (sm_75)** via `sassasm`

The assembler supports 100+ instruction families including:
- Integer ALU: IMAD, IADD3, ISETP, IMNMX
- Logic: LOP3, PRMT, SHF, BFE, BFI
- Float: FADD, FMUL, FFMA, HFMA, MUFU, RRO
- Memory: LD/ST (global, local, shared, sys, constant)
- Control: BRA, BSSY, BAR, BRX, EXIT, WARPSYNC
- Uniform: ULDC, ULOP3, USHF, UIADD3, UIMAD, UPRMT, ULEA
- Vote: VOTEU.ANY, VOTEU.ALL
- 128-bit & 64-bit sys memory
- Extended forms (.X, .HI, .LO, .WIDE, .SX32, .W)

## Integration with sassasm

HASS uses the same `sassasm` backend for assembly, ensuring identical binary output:

```python
from hass import HassAssembler, load_hass_file

program = load_hass_file('kernel.hass')
assembler = HassAssembler(arch='turing', sassasm_path='./sassasm')
assembler.assemble(program, 'kernel.bin')
```

## File Extensions

| Format | Extension |
|--------|-----------|
| SASS (NVIDIA format) | `.sass` |
| HASS (Human-readable) | `.hass` |
| Binary | `.bin` |
| Listing | `.bin.txt` |

## License

Part of the anubis-solver project. MIT + attribution.