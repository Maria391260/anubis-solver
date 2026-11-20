/*
 * sassasm — SASS assembler for NVIDIA GPUs (Kepler, Turing, etc.)
 * Codename: shyleaf
 * Status: UNRELEASED — internal development build
 * 
 * The encoding is derived empirically via the ptxas/nvdisasm oracle:
 * ptxas compiles PTX to cubin, nvdisasm decodes the bytes back to text.
 * Pairing them gives ground truth for each instruction's exact 16-byte encoding.
 * 
 * This is a partial assembler; full coverage requires the format tables
 * from nvdisasm's .data section (recoverable via REA/Ghidra).
 */

#include "sassasm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include <stdint.h>
#include <strings.h>  /* strcasecmp, strncasecmp */

static const char *error_strings[] = {
    [0] = "success",
    [1] = "parse error",
    [2] = "encode error",
    [3] = "unsupported instruction",
    [4] = "register/value out of range",
    [5] = "invalid operand",
};

/* Conservative control words (scheduler-chosen, but these work for dependency-free) */
static const uint8_t CTRL_PLAIN[4] = {0x00, 0xe2, 0x0f, 0x00};
static const uint8_t CTRL_CONST[4] = {0x00, 0xe4, 0x0f, 0x00};

/* ===================== Kepler (sm_35/sm_37) Field Maps ===================== */
/* Based on nvdisasm analysis of Kepler cubins. Opcode bytes are little-endian in file order. */
static const sass_field_map_t kepler_field_maps[] = {
    /* MOV Rd, imm32 */
    {"MOV_IMM", 0x7802, 2, -1, -1, -1, -1, -1, -1,
     "MOV Rd, 0ximm: opcode 02 78, dst at +2, imm32 at +4, ctrl at +12"},

    /* MOV Rd, c[bank][off] */
    {"MOV_CONST", 0x7a02, 2, -1, -1, -1, -1, -1, -1,
     "MOV Rd, c[b][o]: opcode 02 7a, dst at +2, index at +6 (bank<<8 | off/4)"},

    /* IADD3 - integer add with 3 sources */
    {"IADD3", 0x1072, 2, 4, 6, 8, 10, -1, -1,
     "IADD3: opcode 72 10 (or 7a 10 with const), dst+2, src1+4, src2+6, src3+8, flags+10"},

    /* LOP3.LUT - logic op with 3 sources + LUT */
    {"LOP3.LUT", 0x1272, 2, 4, 6, 8, 10, 9, -1,
     "LOP3: opcode 72 12, dst+2, src1+4, src2+6, src3+8, lut+9, flags+10"},

    /* SHF.L/R.U32 - shift funnel */
    {"SHF.L.U32", 0x1978, 2, -1, 8, -1, 9, -1, 4,
     "SHF.L: opcode 78 19, dst+2, shift+4, src1+8, flags+9"},
    {"SHF.R.U32", 0x1978, 2, -1, 8, -1, 9, -1, 4,
     "SHF.R: same opcode, direction in flags byte"},

    /* STG.E - store global */
    {"STG.E", 0x8679, -1, -1, -1, -1, -1, -1, -1,
     "STG.E: opcode 79 86, address from uniform register pair"},

    /* EXIT */
    {"EXIT", 0x794d, -1, -1, -1, -1, -1, -1, -1,
     "EXIT: fixed opcode 4d 79"},

    /* NOP */
    {"NOP", 0x7918, -1, -1, -1, -1, -1, -1, -1,
     "NOP: fixed opcode 18 79"},

    /* IMAD - integer multiply-add */
    {"IMAD", 0x1872, 2, 4, 6, -1, 10, -1, -1,
     "IMAD: opcode 72 18, dst+2, src1+4, src2+6, flags+10"},

    /* IMAD.WIDE.U32 */
    {"IMAD.WIDE.U32", 0x1a72, 2, 4, 6, -1, 10, -1, -1,
     "IMAD.WIDE: opcode 72 1a"},

    /* ISETP - integer compare */
    {"ISETP", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
     "ISETP: opcode 72 1c, writes predicate, src1+4, src2+6"},

    /* BRA - branch */
    {"BRA", 0x8079, -1, -1, -1, -1, -1, -1, -1,
     "BRA: opcode 79 80, target in upper bytes"},

    /* LDS - load shared */
    {"LDS.128", 0x8279, 2, -1, -1, -1, -1, -1, -1,
     "LDS.128: opcode 79 82, dst+2"},

    /* STS - store shared */
    {"STS.128", 0x8479, -1, -1, -1, -1, -1, -1, -1,
     "STS.128: opcode 79 84"},

    /* LDG - load global */
    {"LDG.128", 0x8179, 2, -1, -1, -1, -1, -1, -1,
     "LDG.128: opcode 79 81, dst+2"},

    /* PRMT - permute bytes */
    {"PRMT", 0x2872, 2, 4, 6, 8, -1, -1, -1,
     "PRMT: opcode 72 28, dst+2, src1+4, src2+6, control+8"},

    /* BFE - bit field extract */
    {"BFE.U32", 0x2072, 2, 4, 6, 8, -1, -1, -1,
     "BFE: opcode 72 20, dst+2, src+4, offset+6, width+8"},

    /* BFI - bit field insert */
    {"BFI", 0x2272, 2, 4, 6, 8, -1, -1, -1,
     "BFI: opcode 72 22, dst+2, base+4, insert+6, mask+8"},

    /* SEL - select */
    {"SEL", 0x2472, 2, 4, 6, 8, -1, -1, -1,
     "SEL: opcode 72 24, dst+2, cond+4, true+6, false+8"},

    /* FADD - float add */
    {"FADD", 0x3072, 2, 4, 6, -1, -1, -1, -1,
     "FADD: opcode 72 30, dst+2, src1+4, src2+6"},

    /* FMUL - float mul */
    {"FMUL", 0x3272, 2, 4, 6, -1, -1, -1, -1,
     "FMUL: opcode 72 32, dst+2, src1+4, src2+6"},

    /* FFMA - float fused mul-add */
    {"FFMA", 0x3472, 2, 4, 6, -1, -1, -1, -1,
     "FFMA: opcode 72 34, dst+2, src1+4, src2+6"},

    /* MUFU - math function unit */
    {"MUFU", 0x4072, 2, 4, -1, -1, -1, -1, -1,
     "MUFU: opcode 72 40, dst+2, src+4, func in flags"},

    /* RRO - range reduction */
    {"RRO", 0x4272, 2, 4, 6, -1, -1, -1, -1,
     "RRO: opcode 72 42, dst+2, src1+4, src2+6"},

    /* HMUL - half precision mul */
    {"HMUL", 0x5072, 2, 4, 6, -1, -1, -1, -1,
     "HMUL: opcode 72 50, dst+2, src1+4, src2+6"},

    /* HFMA - half precision fma */
    {"HFMA", 0x5272, 2, 4, 6, -1, -1, -1, -1,
     "HFMA: opcode 72 52, dst+2, src1+4, src2+6"},

    /* S2R - special register to register */
    {"S2R", 0x2e72, 2, 4, -1, -1, -1, -1, -1,
     "S2R: opcode 72 2e, dst+2, src_special+4"},

    /* STL.128 - store local 128-bit */
    {"STL.128", 0x8579, -1, 4, -1, -1, -1, -1, -1,
     "STL.128: opcode 79 85, addr+4"},

    /* STL - store local 32/64-bit */
    {"STL", 0x8779, -1, 4, -1, -1, -1, -1, -1,
     "STL: opcode 79 87, addr+4"},

    /* STL.U8 - store local byte */
    {"STL.U8", 0x8f79, -1, 4, -1, -1, -1, -1, -1,
     "STL.U8: opcode 79 8f, addr+4"},

    /* LDL.128 - load local 128-bit */
    {"LDL.128", 0x8379, 2, 4, -1, -1, -1, -1, -1,
     "LDL.128: opcode 79 83, dst+2, addr+4"},

    /* LDL - load local 32/64-bit */
    {"LDL", 0x8979, 2, 4, -1, -1, -1, -1, -1,
     "LDL: opcode 79 89, dst+2, addr+4"},

    /* LDG.128 - load global 128-bit */
    {"LDG.128", 0x8179, 2, 4, -1, -1, -1, -1, -1,
     "LDG.128: opcode 79 81, dst+2, addr+4"},

    /* LDG - load global 32/64-bit */
    {"LDG", 0x8b79, 2, 4, -1, -1, -1, -1, -1,
     "LDG: opcode 79 8b, dst+2, addr+4"},

    /* IMAD.X - extended IMAD */
    {"IMAD.X", 0x1e72, 2, 4, 6, -1, 10, -1, -1,
     "IMAD.X: opcode 72 1e, dst+2, src1+4, src2+6, flags+10"},

    /* IMAD.MOV.U32 - move variant of IMAD */
    {"IMAD.MOV.U32", 0x1472, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.MOV.U32: opcode 72 14, dst+2, src1+4, src2+6"},

    /* UMOV - uniform register move */
    {"UMOV", 0x2c72, 2, 4, -1, -1, -1, -1, -1,
     "UMOV: opcode 72 2c, dst+2, src+4"},

    /* BMOV - bit move */
    {"BMOV.32.CLEAR", 0x3e72, 2, 4, -1, -1, -1, -1, -1,
     "BMOV.32.CLEAR: opcode 72 3e, dst+2, src+4"},

    /* BSSY - barrier sync */
    {"BSSY", 0x7879, -1, -1, -1, -1, -1, -1, -1,
     "BSSY: opcode 79 78, target in upper bytes"},

    /* ATOMG.E.ADD - atomic global add */
    {"ATOMG.E.ADD", 0x9079, -1, 4, 6, -1, -1, -1, -1,
     "ATOMG: opcode 79 90, addr+4, src+6"},

    /* VOTEU - warp vote any */
    {"VOTEU", 0x4e72, 2, 4, -1, -1, -1, -1, -1,
     "VOTEU: opcode 72 4e, dst+2, src+4"},

    /* FLO - find leading one */
    {"FLO", 0x5472, 2, 4, -1, -1, -1, -1, -1,
     "FLO: opcode 72 54, dst+2, src+4"},

    /* POPC - population count */
    {"POPC", 0x5672, 2, 4, -1, -1, -1, -1, -1,
     "POPC: opcode 72 56, dst+2, src+4"},

    /* SHFL - shuffle */
    {"SHFL", 0x5872, 2, 4, 6, 8, -1, -1, -1,
     "SHFL: opcode 72 58, dst+2, src+4, lane+6, width+8"},

    /* ISETP.GE.U32.AND - compare greater-equal */
    {"ISETP.GE.U32.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
     "ISETP.GE: opcode 72 1c, writes predicate, src1+4, src2+6"},

    /* ISETP.NE.AND - compare not-equal */
    {"ISETP.NE.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
     "ISETP.NE: opcode 72 1c, writes predicate, src1+4, src2+6"},

    /* ISETP.NE.AND.EX - compare not-equal extended */
    {"ISETP.NE.AND.EX", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
     "ISETP.NE.EX: opcode 72 1c, writes predicate, src1+4, src2+6"},

    /* BMOV.32.CLEAR with predicate */
    {"BMOV.32.CLEAR", 0x3e72, 2, 4, -1, -1, -1, -1, -1,
     "BMOV.32.CLEAR: opcode 72 3e, dst+2, src+4"},

    /* ATOMG.E.ADD.STRONG.GPU */
    {"ATOMG.E.ADD.STRONG.GPU", 0x9079, -1, 4, 6, -1, -1, -1, -1,
     "ATOMG.STRONG: opcode 79 90, addr+4, src+6"},

    /* SHF.R.U64 - 64-bit funnel shift right */
    {"SHF.R.U64", 0x1b78, 2, -1, 8, -1, 9, -1, 4,
     "SHF.R.U64: opcode 78 1b, dst+2, shift+4, src+8, flags+9"},

    /* SHF.L.U32.HI - funnel shift left high */
    {"SHF.L.U32.HI", 0x1978, 2, -1, 8, -1, 9, -1, 4,
     "SHF.L.U32.HI: opcode 78 19, dst+2, shift+4, src+8, flags+9"},

    /* SHF.R.U32.HI - funnel shift right high */
    {"SHF.R.U32.HI", 0x1978, 2, -1, 8, -1, 9, -1, 4,
     "SHF.R.U32.HI: opcode 78 19, dst+2, shift+4, src+8, flags+9"},
};

static const int kepler_num_field_maps = sizeof(kepler_field_maps) / sizeof(kepler_field_maps[0]);

/* ===================== Turing (sm_75) Field Maps Extended ===================== */
static const sass_field_map_t turing_field_maps_extended[] = {
    {"MOV_IMM", 0x7802, 2, -1, -1, -1, -1, -1, -1,
     "MOV Rd, imm32"},
    {"MOV_CONST", 0x7a02, 2, -1, -1, -1, -1, -1, -1,
     "MOV Rd, c[b][o]"},
    {"IADD3", 0x1072, 2, 4, 6, 8, 10, -1, -1,
     "IADD3: 0x107a when src2 is constant"},
    {"LOP3.LUT", 0x1272, 2, 4, 6, 8, 10, 9, -1,
     "LOP3: LUT at +9, flags at +10"},
    {"SHF.L.U32", 0x1978, 2, -1, 8, -1, 9, -1, 4,
     "SHF.L/R.U32: direction in flags"},
    {"STG.E", 0x8679, -1, -1, -1, -1, -1, -1, -1,
     "STG.E: address from uniform reg pair"},
    {"EXIT", 0x794d, -1, -1, -1, -1, -1, -1, -1, ""},
    {"NOP", 0x7918, -1, -1, -1, -1, -1, -1, -1, ""},

    /* Basic ALU instructions */
    {"IMAD", 0x1872, 2, 4, 6, -1, 10, -1, -1,
     "IMAD: opcode 72 18, dst+2, src1+4, src2+6, flags+10"},
    {"IMAD.WIDE.U32", 0x1a72, 2, 4, 6, -1, 10, -1, -1,
     "IMAD.WIDE: opcode 72 1a"},
    {"ISETP", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
     "ISETP: opcode 72 1c, writes predicate, src1+4, src2+6"},
    {"FADD", 0x3072, 2, 4, 6, -1, -1, -1, -1,
     "FADD: opcode 72 30, dst+2, src1+4, src2+6"},
    {"FMUL", 0x3272, 2, 4, 6, -1, -1, -1, -1,
     "FMUL: opcode 72 32, dst+2, src1+4, src2+6"},
    {"FFMA", 0x3472, 2, 4, 6, -1, -1, -1, -1,
     "FFMA: opcode 72 34, dst+2, src1+4, src2+6"},
    {"MUFU", 0x4072, 2, 4, -1, -1, -1, -1, -1,
     "MUFU: opcode 72 40, dst+2, src+4, func in flags"},
    {"RRO", 0x4272, 2, 4, 6, -1, -1, -1, -1,
     "RRO: opcode 72 42, dst+2, src1+4, src2+6"},
    {"HMUL", 0x5072, 2, 4, 6, -1, -1, -1, -1,
     "HMUL: opcode 72 50, dst+2, src1+4, src2+6"},
    {"HFMA", 0x5272, 2, 4, 6, -1, -1, -1, -1,
     "HFMA: opcode 72 52, dst+2, src1+4, src2+6"},
    {"S2R", 0x2e72, 2, 4, -1, -1, -1, -1, -1,
     "S2R: opcode 72 2e, dst+2, src_special+4"},
    {"STL.128", 0x8579, -1, 4, -1, -1, -1, -1, -1,
     "STL.128: opcode 79 85, addr+4"},
    {"STL", 0x8779, -1, 4, -1, -1, -1, -1, -1,
     "STL: opcode 79 87, addr+4"},
    {"STL.U8", 0x8f79, -1, 4, -1, -1, -1, -1, -1,
     "STL.U8: opcode 79 8f, addr+4"},
    {"LDL.128", 0x8379, 2, 4, -1, -1, -1, -1, -1,
     "LDL.128: opcode 79 83, dst+2, addr+4"},
    {"LDL", 0x8979, 2, 4, -1, -1, -1, -1, -1,
     "LDL: opcode 79 89, dst+2, addr+4"},
    {"LDG.128", 0x8179, 2, 4, -1, -1, -1, -1, -1,
     "LDG.128: opcode 79 81, dst+2, addr+4"},
    {"LDG", 0x8b79, 2, 4, -1, -1, -1, -1, -1,
     "LDG: opcode 79 8b, dst+2, addr+4"},
    {"IMAD.X", 0x1e72, 2, 4, 6, -1, 10, -1, -1,
     "IMAD.X: opcode 72 1e, dst+2, src1+4, src2+6, flags+10"},
    {"IMAD.MOV.U32", 0x1472, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.MOV.U32: opcode 72 14, dst+2, src1+4, src2+6"},
    {"UMOV", 0x2c72, 2, 4, -1, -1, -1, -1, -1,
     "UMOV: opcode 72 2c, dst+2, src+4"},
    {"BMOV.32.CLEAR", 0x3e72, 2, 4, -1, -1, -1, -1, -1,
     "BMOV.32.CLEAR: opcode 72 3e, dst+2, src+4"},
    {"BSSY", 0x7879, -1, -1, -1, -1, -1, -1, -1,
     "BSSY: opcode 79 78, target in upper bytes"},
    {"ATOMG.E.ADD", 0x9079, -1, 4, 6, -1, -1, -1, -1,
     "ATOMG: opcode 79 90, addr+4, src+6"},
    {"VOTEU", 0x4e72, 2, 4, -1, -1, -1, -1, -1,
     "VOTEU: opcode 72 4e, dst+2, src+4"},
    {"FLO", 0x5472, 2, 4, -1, -1, -1, -1, -1,
     "FLO: opcode 72 54, dst+2, src+4"},
    {"POPC", 0x5672, 2, 4, -1, -1, -1, -1, -1,
     "POPC: opcode 72 56, dst+2, src+4"},
    {"SHFL", 0x5872, 2, 4, 6, 8, -1, -1, -1,
     "SHFL: opcode 72 58, dst+2, src+4, lane+6, width+8"},
    {"ISETP.GE.U32.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
     "ISETP.GE: opcode 72 1c, writes predicate, src1+4, src2+6"},
    {"ISETP.NE.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
     "ISETP.NE: opcode 72 1c, writes predicate, src1+4, src2+6"},
    {"SHF.R.U64", 0x1b78, 2, -1, 8, -1, 9, -1, 4,
     "SHF.R.U64: opcode 78 1b, dst+2, shift+4, src+8, flags+9"},
    {"SHF.R.U32.HI", 0x1978, 2, -1, 8, -1, 9, -1, 4,
     "SHF.R.U32.HI: opcode 78 19, dst+2, shift+4, src+8, flags+9"},
    {"SHF.L.U32.HI", 0x1978, 2, -1, 8, -1, 9, -1, 4,
     "SHF.L.U32.HI: opcode 78 19, dst+2, shift+4, src+8, flags+9"},
    {"SHF.L.U64.HI", 0x1b78, 2, -1, 8, -1, 9, -1, 4,
     "SHF.L.U64.HI: opcode 78 1b, dst+2, shift+4, src+8, flags+9"},
    {"SHF.L.W.U32.HI", 0x1d78, 2, -1, 8, -1, 9, -1, 4,
     "SHF.L.W.U32.HI: opcode 78 1d, dst+2, shift+4, src+8, flags+9"},
    {"PRMT", 0x2872, 2, 4, 6, 8, -1, -1, -1,
     "PRMT: opcode 72 28, dst+2, src1+4, src2+6, control+8"},
    {"ATOMG.E.ADD.STRONG.GPU", 0x9079, -1, 4, 6, -1, -1, -1, -1,
     "ATOMG.STRONG: opcode 79 90, addr+4, src+6"},

    /* Additional instructions from kernels */
    {"IMAD.IADD", 0x1672, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.IADD: opcode 72 16, dst+2, src1+4, src2+6"},
    {"IMAD.SHL.U32", 0x1e72, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.SHL.U32: opcode 72 1e, dst+2, src1+4, src2+6, shift in src2"},
    {"LDC", 0x2a72, 2, 4, -1, -1, -1, -1, -1,
     "LDC: opcode 72 2a, dst+2, addr+4"},
    {"LDL.64", 0x8d79, 2, 4, -1, -1, -1, -1, -1,
     "LDL.64: opcode 79 8d, dst+2, addr+4"},
    {"STL.64", 0x8f79, -1, 4, -1, -1, -1, -1, -1,
     "STL.64: opcode 79 8f, addr+4"},
    {"STG.E.SYS", 0x9279, -1, 4, 6, -1, -1, -1, -1,
     "STG.E.SYS: opcode 79 92, addr+4, src+6"},
    {"LDG.E.128.SYS", 0x9179, 2, 4, -1, -1, -1, -1, -1,
     "LDG.E.128.SYS: opcode 79 91, dst+2, addr+4"},
    {"SHF.R.S32.HI", 0x1b78, 2, -1, 8, -1, 9, -1, 4,
     "SHF.R.S32.HI: opcode 78 1b, dst+2, shift+4, src+8, flags+9"},
    {"SHFL.IDX", 0x5a72, 2, 4, 6, 8, -1, -1, -1,
     "SHFL.IDX: opcode 72 5a, dst+2, src+4, lane+6, width+8"},
    {"IMNMX.U32", 0x4472, 2, 4, 6, -1, -1, -1, -1,
     "IMNMX.U32: opcode 72 44, dst+2, src1+4, src2+6"},
    {"CS2R", 0x3c72, 2, 4, -1, -1, -1, -1, -1,
     "CS2R: opcode 72 3c, dst+2, src+4"},
    {"LEA.HI", 0x6072, 2, 4, 6, 8, -1, -1, -1,
     "LEA.HI: opcode 72 60, dst+2, base+4, index+6, scale+8"},
    {"BRA", 0x8079, -1, -1, -1, -1, -1, -1, -1,
     "BRA: opcode 79 80, target in upper bytes"},
    {"BRX", 0x8179, -1, -1, -1, -1, -1, -1, -1,
     "BRX: opcode 79 81, target in upper bytes"},
    {"BSSY", 0x7879, -1, -1, -1, -1, -1, -1, -1,
     "BSSY: opcode 79 78, target in upper bytes"},
    {"BSYNC", 0x7979, -1, -1, -1, -1, -1, -1, -1,
     "BSYNC: opcode 79 79, target in upper bytes"},

    /* Uniform register instructions */
    {"ULDC.64", 0x2e72, 2, 4, -1, -1, -1, -1, -1,
     "ULDC.64: opcode 72 2e, URDst+2, c[bank][off]+4"},
    {"ULOP3.LUT", 0x1272, 2, 4, 6, 8, 10, 9, -1,
     "ULOP3: opcode 72 12, URDst+2, URsrc1+4, URsrc2+6, URsrc3+8, lut+9, flags+10"},
    {"USHF.L.U32", 0x1978, 2, -1, 8, -1, 9, -1, 4,
     "USHF.L.U32: opcode 78 19, URDst+2, shift+4, URsrc+8, flags+9"},
    {"USHF.R.U32.HI", 0x1978, 2, -1, 8, -1, 9, -1, 4,
     "USHF.R.U32.HI: opcode 78 19, URDst+2, shift+4, URsrc+8, flags+9"},
    {"USHF.L.U64.HI", 0x1b78, 2, -1, 8, -1, 9, -1, 4,
     "USHF.L.U64.HI: opcode 78 1b, URDst+2, shift+4, URsrc+8, flags+9"},
    {"USHF.R.U64", 0x1b78, 2, -1, 8, -1, 9, -1, 4,
     "USHF.R.U64: opcode 78 1b, URDst+2, shift+4, URsrc+8, flags+9"},
    {"UIADD3", 0x1072, 2, 4, 6, 8, 10, -1, -1,
     "UIADD3: opcode 72 10, URDst+2, URsrc1+4, URsrc2+6, URsrc3+8, flags+10"},
    {"UIADD3.X", 0x1072, 2, 4, 6, 8, 10, -1, -1,
     "UIADD3.X: opcode 72 10, URDst+2, URsrc1+4, URsrc2+6, URsrc3+8, flags+10"},
    {"UMOV", 0x2c72, 2, 4, -1, -1, -1, -1, -1,
     "UMOV: opcode 72 2c, URDst+2, URsrc+4"},

    /* Extended forms with predicates */
    {"IADD3.X", 0x1072, 2, 4, 6, 8, 10, -1, -1,
     "IADD3.X: opcode 72 10, dst+2, src1+4, src2+6, src3+8, flags+10"},
    {"IADD3.X", 0x107a, 2, 4, 6, 8, 10, -1, -1,
     "IADD3.X (const): opcode 72 1a, dst+2, src1+4, src2+6, src3+8, flags+10"},
    {"IMAD.X", 0x1e72, 2, 4, 6, -1, 10, -1, -1,
     "IMAD.X: opcode 72 1e, dst+2, src1+4, src2+6, flags+10"},
    {"IMAD.WIDE.U32.X", 0x1a72, 2, 4, 6, -1, 10, -1, -1,
     "IMAD.WIDE.X: opcode 72 1a, dst+2, src1+4, src2+6, flags+10"},
    {"IMAD.WIDE.U32.X", 0x1e72, 2, 4, 6, -1, 10, -1, -1,
     "IMAD.WIDE.X (alt): opcode 72 1e, dst+2, src1+4, src2+6, flags+10"},
    {"IMAD.X.X", 0x1e72, 2, 4, 6, -1, 10, -1, -1,
     "IMAD.X.X: opcode 72 1e, dst+2, src1+4, src2+6, flags+10"},

    /* 64-bit sys memory */
    {"LDG.E.64.SYS", 0x9379, 2, 4, -1, -1, -1, -1, -1,
     "LDG.E.64.SYS: opcode 79 93, dst+2, addr+4"},
    {"STG.E.64.SYS", 0x9479, -1, 4, 6, -1, -1, -1, -1,
     "STG.E.64.SYS: opcode 79 94, addr+4, src+6"},
    {"LDS.U.U8", 0x8e79, 2, 4, -1, -1, -1, -1, -1,
     "LDS.U.U8: opcode 79 8e, dst+2, addr+4"},

    /* More LDS/STS variants */
    {"LDS.64", 0x8c79, 2, 4, -1, -1, -1, -1, -1,
     "LDS.64: opcode 79 8c, dst+2, addr+4"},
    {"STS.64", 0x8e79, -1, 4, -1, -1, -1, -1, -1,
     "STS.64: opcode 79 8e, addr+4"},
    {"LDS.U.64", 0x8e79, 2, 4, -1, -1, -1, -1, -1,
     "LDS.U.64: opcode 79 8e, dst+2, addr+4"},
    {"STS.U.64", 0x9079, -1, 4, -1, -1, -1, -1, -1,
     "STS.U.64: opcode 79 90, addr+4"},

    /* FLO variants */
    {"FLO.U32", 0x5472, 2, 4, -1, -1, -1, -1, -1,
     "FLO.U32: opcode 72 54, dst+2, src+4"},

    /* More ISETP variants */
    {"ISETP.GT.U32.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
     "ISETP.GT: opcode 72 1c, writes predicate, src1+4, src2+6"},
    {"ISETP.LT.U32.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
     "ISETP.LT: opcode 72 1c, writes predicate, src1+4, src2+6"},
    {"ISETP.EQ.U32.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
     "ISETP.EQ: opcode 72 1c, writes predicate, src1+4, src2+6"},
    {"ISETP.NE.U32.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
     "ISETP.NE: opcode 72 1c, writes predicate, src1+4, src2+6"},
    {"ISETP.GE.U32.AND.EX", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
     "ISETP.GE.EX: opcode 72 1c, writes predicate, src1+4, src2+6"},
    {"ISETP.NE.AND.EX", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
     "ISETP.NE.EX: opcode 72 1c, writes predicate, src1+4, src2+6"},

    /* IMNMX variants */
    {"IMNMX.U32", 0x4472, 2, 4, 6, -1, -1, -1, -1,
     "IMNMX.U32: opcode 72 44, dst+2, src1+4, src2+6"},
    {"IMNMX.S32", 0x4672, 2, 4, 6, -1, -1, -1, -1,
     "IMNMX.S32: opcode 72 46, dst+2, src1+4, src2+6"},

    /* LEA variants */
    {"LEA.HI", 0x6072, 2, 4, 6, 8, -1, -1, -1,
     "LEA.HI: opcode 72 60, dst+2, base+4, index+6, scale+8"},
    {"LEA.LO", 0x6272, 2, 4, 6, 8, -1, -1, -1,
     "LEA.LO: opcode 72 62, dst+2, base+4, index+6, scale+8"},

    /* BAR/MEMBAR */
    {"BAR.SYNC", 0x7a79, -1, -1, -1, -1, -1, -1, -1,
     "BAR.SYNC: opcode 79 7a"},
    {"MEMBAR.SYS", 0x7b79, -1, -1, -1, -1, -1, -1, -1,
     "MEMBAR.SYS: opcode 79 7b"},
    {"MEMBAR.CTA", 0x7c79, -1, -1, -1, -1, -1, -1, -1,
     "MEMBAR.CTA: opcode 79 7c"},

    /* SEL variants */
    {"SEL", 0x2472, 2, 4, 6, 8, -1, -1, -1,
     "SEL: opcode 72 24, dst+2, cond+4, true+6, false+8"},
    {"SEL.EQ", 0x2472, 2, 4, 6, 8, -1, -1, -1,
     "SEL.EQ: opcode 72 24, dst+2, cond+4, true+6, false+8"},
    {"SEL.NE", 0x2472, 2, 4, 6, 8, -1, -1, -1,
     "SEL.NE: opcode 72 24, dst+2, cond+4, true+6, false+8"},

    /* BFE/BFI variants */
    {"BFE.U32", 0x2072, 2, 4, 6, 8, -1, -1, -1,
     "BFE.U32: opcode 72 20, dst+2, src+4, offset+6, width+8"},
    {"BFE.S32", 0x2172, 2, 4, 6, 8, -1, -1, -1,
     "BFE.S32: opcode 72 21, dst+2, src+4, offset+6, width+8"},
    {"BFI", 0x2272, 2, 4, 6, 8, -1, -1, -1,
     "BFI: opcode 72 22, dst+2, base+4, insert+6, mask+8"},

    /* PSETP (predicate set) */
    {"PSETP.AND", 0x3a72, -1, 4, 6, -1, -1, -1, -1,
     "PSETP.AND: opcode 72 3a, writes predicate, src1+4, src2+6"},
    {"PSETP.OR", 0x3a72, -1, 4, 6, -1, -1, -1, -1,
     "PSETP.OR: opcode 72 3a, writes predicate, src1+4, src2+6"},
    {"PSETP.XOR", 0x3a72, -1, 4, 6, -1, -1, -1, -1,
     "PSETP.XOR: opcode 72 3a, writes predicate, src1+4, src2+6"},

    /* VOTE variants */
    {"VOTE.ANY", 0x4e72, 2, 4, -1, -1, -1, -1, -1,
     "VOTE.ANY: opcode 72 4e, dst+2, src+4"},
    {"VOTE.ALL", 0x4f72, 2, 4, -1, -1, -1, -1, -1,
     "VOTE.ALL: opcode 72 4f, dst+2, src+4"},
    {"VOTE.BALLOT", 0x5072, 2, 4, 6, -1, -1, -1, -1,
     "VOTE.BALLOT: opcode 72 50, dst+2, src+4, ballot+6"},

    /* FCHK (float check) */
    {"FCHK", 0x5c72, 2, 4, -1, -1, -1, -1, -1,
     "FCHK: opcode 72 5c, dst+2, src+4"},
};
static const int turing_num_field_maps_extended = sizeof(turing_field_maps_extended) / sizeof(turing_field_maps_extended[0]);

/* ===================== Turing (sm_75) Field Maps ===================== */
/* These match the Python FIELD_MAPS */
static const sass_field_map_t turing_field_maps[] = {
    {"MOV_IMM", 0x7802, 2, -1, -1, -1, -1, -1, -1,
     "MOV Rd, imm32"},
    {"MOV_CONST", 0x7a02, 2, -1, -1, -1, -1, -1, -1,
     "MOV Rd, c[b][o]"},
    {"IADD3", 0x1072, 2, 4, 6, 8, 10, -1, -1,
     "IADD3: 0x107a when src2 is constant"},
    {"LOP3.LUT", 0x1272, 2, 4, 6, 8, 10, 9, -1,
     "LOP3: LUT at +9, flags at +10"},
    {"SHF.L.U32", 0x1978, 2, -1, 8, -1, 9, -1, 4,
     "SHF.L/R.U32: direction in flags"},
    {"STG.E", 0x8679, -1, -1, -1, -1, -1, -1, -1,
     "STG.E: address from uniform reg pair"},
    {"EXIT", 0x794d, -1, -1, -1, -1, -1, -1, -1, ""},
    {"NOP", 0x7918, -1, -1, -1, -1, -1, -1, -1, ""},
};
static const int turing_num_field_maps = sizeof(turing_field_maps) / sizeof(turing_field_maps[0]);

/* ===================== Helper Functions ===================== */

static void write_le16(uint8_t *buf, int offset, uint16_t val) {
    buf[offset] = val & 0xFF;
    buf[offset + 1] = (val >> 8) & 0xFF;
}

static void write_le32(uint8_t *buf, int offset, uint32_t val) {
    buf[offset] = val & 0xFF;
    buf[offset + 1] = (val >> 8) & 0xFF;
    buf[offset + 2] = (val >> 16) & 0xFF;
    buf[offset + 3] = (val >> 24) & 0xFF;
}

static void write_le64(uint8_t *buf, int offset, uint64_t val) {
    write_le32(buf, offset, val & 0xFFFFFFFF);
    write_le32(buf, offset + 4, (val >> 32) & 0xFFFFFFFF);
}

static const sass_field_map_t *find_field_map(sass_context_t *ctx, const char *mnemonic) {
    for (int i = 0; i < ctx->num_field_maps; i++) {
        if (strcasecmp(ctx->field_maps[i].name, mnemonic) == 0) {
            return &ctx->field_maps[i];
        }
    }
    return NULL;
}

/* ===================== Register Parsing ===================== */

int sass_reg_from_string(const char *s, sass_reg_type_t *type, int *index) {
    if (!s) return SASS_ERR_PARSE;

    /* Skip leading whitespace */
    while (*s && isspace(*s)) s++;

    if (s[0] == 'R' || s[0] == 'r') {
        *type = SASS_REG_GPR;
        *index = atoi(s + 1);
        if (*index < 0 || *index > 255) return SASS_ERR_RANGE;
        return SASS_OK;
    }
    if (s[0] == 'P' || s[0] == 'p') {
        *type = SASS_REG_PRED;
        *index = atoi(s + 1);
        if (*index < 0 || *index > 7) return SASS_ERR_RANGE;
        return SASS_OK;
    }
    if (strncasecmp(s, "CC", 2) == 0) {
        *type = SASS_REG_CCB;
        *index = atoi(s + 2);
        if (*index < 0 || *index > 3) return SASS_ERR_RANGE;
        return SASS_OK;
    }
    if (strncasecmp(s, "UR", 2) == 0) {
        *type = SASS_REG_UNIFORM;
        *index = atoi(s + 2);
        return SASS_OK;
    }
    if (strncasecmp(s, "SR_", 3) == 0) {
        /* Special registers: SR_CTAID.X, SR_TID.X, etc. */
        *type = SASS_REG_SPECIAL;
        if (strcasecmp(s + 3, "CTAID.X") == 0) *index = 2;
        else if (strcasecmp(s + 3, "TID.X") == 0) *index = 3;
        else if (strcasecmp(s + 3, "CTAID.Y") == 0) *index = 4;
        else if (strcasecmp(s + 3, "TID.Y") == 0) *index = 5;
        else if (strcasecmp(s + 3, "CTAID.Z") == 0) *index = 6;
        else if (strcasecmp(s + 3, "TID.Z") == 0) *index = 7;
        else if (strcasecmp(s + 3, "LANEID") == 0) *index = 8;
        else if (strcasecmp(s + 3, "WARPID") == 0) *index = 9;
        else if (strcasecmp(s + 3, "NCTAID.X") == 0) *index = 10;
        else if (strcasecmp(s + 3, "NCTAID.Y") == 0) *index = 11;
        else if (strcasecmp(s + 3, "NCTAID.Z") == 0) *index = 12;
        else *index = 13;  /* unknown SR */
        return SASS_OK;
    }
    if (strcasecmp(s, "RZ") == 0) {
        *type = SASS_REG_SPECIAL;
        *index = 0;  /* RZ = zero register */
        return SASS_OK;
    }
    if (strcasecmp(s, "SP") == 0) {
        *type = SASS_REG_SPECIAL;
        *index = 1;  /* stack pointer */
        return SASS_OK;
    }
    if (strcasecmp(s, "SRZ") == 0) {
        *type = SASS_REG_SPECIAL;
        *index = 14;  /* SRZ = zero special register */
        return SASS_OK;
    }
    if (s[0] == 'B' || s[0] == 'b') {
        *type = SASS_REG_SPECIAL;
        *index = 16 + atoi(s + 1);  /* B0-B15 -> 16-31 */
        if (*index < 16 || *index > 31) return SASS_ERR_RANGE;
        return SASS_OK;
    }

    return SASS_ERR_PARSE;
}

const char *sass_reg_to_string(sass_reg_type_t type, int index) {
    static char buf[16];
    switch (type) {
        case SASS_REG_GPR: snprintf(buf, sizeof(buf), "R%d", index); break;
        case SASS_REG_PRED: snprintf(buf, sizeof(buf), "P%d", index); break;
        case SASS_REG_CCB: snprintf(buf, sizeof(buf), "CC%d", index); break;
        case SASS_REG_UNIFORM: snprintf(buf, sizeof(buf), "UR%d", index); break;
        case SASS_REG_SPECIAL:
            if (index == 0) snprintf(buf, sizeof(buf), "RZ");
            else if (index == 1) snprintf(buf, sizeof(buf), "SP");
            else snprintf(buf, sizeof(buf), "SPEC%d", index);
            break;
    }
    return buf;
}

/* ===================== Control Word ===================== */

uint32_t sass_ctrl_plain(void) {
    return *(uint32_t *)CTRL_PLAIN;
}

uint32_t sass_ctrl_const(void) {
    return *(uint32_t *)CTRL_CONST;
}

uint32_t sass_ctrl_custom(int stall, int yield, int rbar, int wbar, int reuse) {
    /* Control word format (empirical):
     * Byte 0: stall count (0-15)
     * Byte 1: yield | read_barrier | write_barrier | reuse flags
     * Byte 2: 0x0f (fixed)
     * Byte 3: 0x00 (fixed)
     */
    uint32_t ctrl = 0;
    ctrl |= (stall & 0xF);
    ctrl |= ((yield & 1) << 4);
    ctrl |= ((rbar & 1) << 5);
    ctrl |= ((wbar & 1) << 6);
    ctrl |= ((reuse & 1) << 7);
    ctrl |= 0x0F00;  /* byte 2 = 0x0f */
    return ctrl;
}

/* Control word with predication:
 * Byte 0: stall count (0-15)
 * Byte 1: predicate register (0-7) | pred_negated << 3 | yield << 4 | rbar << 5 | wbar << 6 | reuse << 7
 * Byte 2: 0x0f (fixed)
 * Byte 3: 0x00 (fixed)
 */
uint32_t sass_ctrl_predicated(int stall, int yield, int rbar, int wbar, int reuse, 
                               int pred_reg, int pred_negated) {
    uint32_t ctrl = 0;
    ctrl |= (stall & 0xF);
    ctrl |= ((pred_reg & 0x7) << 0);       /* bits 0-2: predicate register */
    ctrl |= ((pred_negated & 1) << 3);     /* bit 3: predicate negated */
    ctrl |= ((yield & 1) << 4);
    ctrl |= ((rbar & 1) << 5);
    ctrl |= ((wbar & 1) << 6);
    ctrl |= ((reuse & 1) << 7);
    ctrl |= 0x0F00;  /* byte 2 = 0x0f */
    return ctrl;
}

/* ===================== Instruction Parsing ===================== */

static char *strdup_trimmed(const char *s) {
    while (*s && isspace(*s)) s++;
    const char *e = s + strlen(s);
    while (e > s && isspace(e[-1])) e--;
    char *r = malloc(e - s + 1);
    if (!r) return NULL;
    memcpy(r, s, e - s);
    r[e - s] = '\0';
    return r;
}

static int parse_operand(const char *tok, sass_operand_t *op) {
    memset(op, 0, sizeof(*op));
    op->cbank = -1;
    op->coffset = -1;

    /* Check for predicate register with negation: @!P0 or @P0 */
    if (tok[0] == '@') {
        const char *p = tok + 1;
        int neg = 0;
        if (*p == '!') {
            neg = 1;
            p++;
        }
        int idx = atoi(p + 1);
        if (idx < 0 || idx > 7) return SASS_ERR_RANGE;
        op->type = SASS_REG_PRED;
        op->index = idx;
        op->is_negated = neg;
        return SASS_OK;
    }

    /* Check for .reuse suffix */
    char *reuse_pos = strstr(tok, ".reuse");
    int has_reuse = 0;
    char tok_buf[128];
    strncpy(tok_buf, tok, sizeof(tok_buf) - 1);
    tok_buf[sizeof(tok_buf) - 1] = '\0';
    if (reuse_pos) {
        has_reuse = 1;
        *reuse_pos = '\0';
    }

    /* Check for branch target: `(.L_x_123) */
    if (tok[0] == '`') {
        op->type = SASS_REG_SPECIAL;
        op->index = 15;  /* branch target marker */
        op->has_imm = 1;
        op->imm_value = 0;  /* placeholder for branch offset */
        op->is_branch_target = 1;
        op->is_reuse = has_reuse;
        return SASS_OK;
    }

    /* Check for memory operand: [R1] or [R1+0x10] */
    if (tok[0] == '[') {
        op->type = SASS_REG_GPR;  /* base register */
        const char *p = tok + 1;
        int base_reg = -1;
        int64_t offset = 0;
        
        /* Parse base register */
        if (p[0] == 'R' || p[0] == 'r') {
            base_reg = atoi(p + 1);
            while (*p && *p != ']' && *p != '+' && *p != '-') p++;
        }
        
        /* Parse optional offset */
        if (*p == '+' || *p == '-') {
            int sign = (*p == '+') ? 1 : -1;
            p++;
            offset = sign * strtoll(p, NULL, 0);
        }
        
        if (base_reg >= 0 && base_reg <= 255) {
            op->index = base_reg;
            op->has_imm = 1;
            op->imm_value = offset;
            op->is_memory = 1;
            op->is_reuse = has_reuse;
            return SASS_OK;
        }
        return SASS_ERR_PARSE;
    }

    /* Check for c[bank][offset] */
    if (tolower(tok[0]) == 'c' && tok[1] == '[') {
        op->type = SASS_REG_GPR;  /* destination is GPR */
        int bank = 0, off = 0;
        const char *p = tok + 2;
        bank = strtol(p, (char **)&p, 0);
        if (*p == ']') p++;
        if (*p == '[') p++;
        off = strtol(p, NULL, 0);
        op->cbank = bank;
        op->coffset = off;
        op->has_imm = 1;
        op->imm_value = off;
        op->is_reuse = has_reuse;
        return SASS_OK;
    }

    /* Check for immediate (hex, decimal, octal) */
    if (tok[0] == '0' && (tok[1] == 'x' || tok[1] == 'X')) {
        op->has_imm = 1;
        op->imm_value = strtoull(tok, NULL, 0);
        op->is_reuse = has_reuse;
        return SASS_OK;
    }
    if (isdigit(tok[0]) || tok[0] == '-') {
        op->has_imm = 1;
        op->imm_value = strtoull(tok, NULL, 0);
        op->is_reuse = has_reuse;
        return SASS_OK;
    }

    /* Register */
    int rc = sass_reg_from_string(tok, &op->type, &op->index);
    if (rc == SASS_OK) {
        op->is_reuse = has_reuse;
    }
    return rc;
}

int sass_parse_line(sass_context_t *ctx, const char *line, sass_instruction_t *insn) {
    if (!line || !insn) return SASS_ERR_PARSE;
    memset(insn, 0, sizeof(*insn));

    char *trimmed = strdup_trimmed(line);
    if (!trimmed) return SASS_ERR_PARSE;

    /* Skip comments and empty lines */
    if (trimmed[0] == '\0' || trimmed[0] == '/' || trimmed[0] == '#' || trimmed[0] == ';') {
        free(trimmed);
        return SASS_ERR_PARSE;  /* caller should skip these */
    }

    /* Remove trailing semicolon */
    char *semi = strrchr(trimmed, ';');
    if (semi) *semi = '\0';

    /* Check for control code prefix: --:-:-:Y:0 */
    if (strncmp(trimmed, "--:", 3) == 0) {
        /* Parse control code - simplified for now */
        insn->has_control_code = 1;
        insn->control_code = sass_ctrl_plain();
        /* Skip to after control code */
        char *space = strchr(trimmed, ' ');
        if (space) {
            memmove(trimmed, space + 1, strlen(space));
        }
    }

    /* Check for predication prefix: @P0 or @!P0 */
    if (trimmed[0] == '@') {
        char *p = trimmed + 1;
        int negated = 0;
        if (*p == '!') {
            negated = 1;
            p++;
        }
        if (*p == 'P' || *p == 'p') {
            int reg = atoi(p + 1);
            if (reg >= 0 && reg <= 7) {
                insn->has_predicate = 1;
                insn->predicate_reg = reg;
                insn->predicate_negated = negated;
                /* Skip past the predicate */
                while (*p && !isspace(*p)) p++;
            }
        }
        /* Skip whitespace after predicate */
        while (*p && isspace(*p)) p++;
        /* Move trimmed to start after predicate */
        memmove(trimmed, p, strlen(p) + 1);
    }

    /* Tokenize: split by spaces and commas */
    char *tokens[16];
    int ntokens = 0;
    char *p = trimmed;
    while (*p && ntokens < 16) {
        while (*p && isspace(*p)) p++;
        if (!*p) break;
        char *start = p;
        while (*p && !isspace(*p) && *p != ',') p++;
        char saved = *p;
        *p = '\0';
        tokens[ntokens++] = start;
        if (saved) p++;
    }

    if (ntokens == 0) {
        free(trimmed);
        return SASS_ERR_PARSE;
    }

    /* First token is mnemonic */
    strncpy(insn->mnemonic, tokens[0], sizeof(insn->mnemonic) - 1);
    insn->mnemonic[sizeof(insn->mnemonic) - 1] = '\0';

    /* Parse operands */
    for (int i = 1; i < ntokens && insn->num_operands < 8; i++) {
        if (tokens[i][0] == '\0') continue;
        int rc = parse_operand(tokens[i], &insn->operands[insn->num_operands]);
        if (rc != SASS_OK) {
            free(trimmed);
            return rc;
        }
        insn->num_operands++;
    }

    free(trimmed);
    return SASS_OK;
}

/* ===================== Instruction Encoding ===================== */

static int encode_mov_imm(sass_context_t *ctx, const sass_instruction_t *insn, sass_encoded_t *out) {
    if (insn->num_operands < 2) return SASS_ERR_PARSE;

    const sass_operand_t *dst = &insn->operands[0];
    const sass_operand_t *src = &insn->operands[1];

    if (dst->type != SASS_REG_GPR) return SASS_ERR_INVALID_OP;
    if (dst->index < 0 || dst->index > 255) return SASS_ERR_RANGE;

    uint8_t *b = out->bytes;
    memset(b, 0, SASS_INSN_SIZE);

    /* Opcode: 02 78 (little-endian: 78 02) */
    write_le16(b, 0, 0x7802);
    /* Destination register at offset 2 */
    b[2] = dst->index & 0xFF;
    b[3] = 0x00;  /* high byte of dst (always 0 for GPR) */

    /* Immediate at offset 4 (32-bit) */
    write_le32(b, 4, src->imm_value & 0xFFFFFFFF);

    /* Bytes 8-11: padding */
    b[8] = 0x00;
    b[9] = 0x0F;
    b[10] = 0x00;
    b[11] = 0x00;

    /* Control word at offset 12 */
    uint32_t ctrl = src->has_imm ? sass_ctrl_plain() : sass_ctrl_const();
    write_le32(b, 12, ctrl);

    return SASS_OK;
}

/* Encode branch instructions (BRA, BSSY, BRX, BAR.SYNC) with backtick targets */
static int encode_branch(sass_context_t *ctx, const sass_instruction_t *insn, sass_encoded_t *out) {
    uint8_t *b = out->bytes;
    memset(b, 0, SASS_INSN_SIZE);

    uint16_t opcode = 0;
    if (strcasecmp(insn->mnemonic, "BRA") == 0) opcode = 0x8079;
    else if (strcasecmp(insn->mnemonic, "BSSY") == 0) opcode = 0x7879;
    else if (strcasecmp(insn->mnemonic, "BRX") == 0) opcode = 0x8179;
    else if (strcasecmp(insn->mnemonic, "BAR.SYNC") == 0) opcode = 0x7a79;
    else return SASS_ERR_UNSUPPORTED;

    write_le16(b, 0, opcode);

    /* Branch target is in upper bytes (offset 4-15) */
    /* For now, use placeholder - real offset would need label resolution */
    for (int i = 0; i < insn->num_operands; i++) {
        const sass_operand_t *op = &insn->operands[i];
        if (op->is_branch_target) {
            /* Branch offset would go here - placeholder for now */
            break;
        }
    }

    /* Control word at offset 12 */
    if (insn->has_predicate) {
        write_le32(b, 12, sass_ctrl_predicated(0, 0, 0, 0, 0, 
                                                 insn->predicate_reg, 
                                                 insn->predicate_negated));
    } else {
        write_le32(b, 12, sass_ctrl_plain());
    }

    return SASS_OK;
}

static int encode_mov_const(sass_context_t *ctx, const sass_instruction_t *insn, sass_encoded_t *out) {
    if (insn->num_operands < 2) return SASS_ERR_PARSE;

    const sass_operand_t *dst = &insn->operands[0];
    const sass_operand_t *src = &insn->operands[1];

    if (dst->type != SASS_REG_GPR) return SASS_ERR_INVALID_OP;
    if (dst->index < 0 || dst->index > 255) return SASS_ERR_RANGE;

    uint8_t *b = out->bytes;
    memset(b, 0, SASS_INSN_SIZE);

    /* Opcode: 02 7A (little-endian: 7A 02) */
    write_le16(b, 0, 0x7A02);
    /* Destination register at offset 2 */
    b[2] = dst->index & 0xFF;
    b[3] = 0x00;

    /* Byte 4: 0x00 */
    b[4] = 0x00;

    /* Constant index at offset 5: (bank << 8) | (offset / 4) - matches Python */
    uint16_t idx = (src->cbank << 8) | (src->coffset / 4);
    write_le16(b, 5, idx);

    /* Bytes 7-11: padding (00 00 0f 00 00) */
    b[7] = 0x00;
    b[8] = 0x00;
    b[9] = 0x0F;
    b[10] = 0x00;
    b[11] = 0x00;

    /* Control word at offset 12 */
    write_le32(b, 12, sass_ctrl_const());

    return SASS_OK;
}

static int encode_generic(sass_context_t *ctx, const sass_instruction_t *insn,
                          const sass_field_map_t *fm, sass_encoded_t *out) {
    uint8_t *b = out->bytes;
    memset(b, 0, SASS_INSN_SIZE);

    /* Opcode (2 bytes, little-endian) */
    write_le16(b, 0, fm->opcode);

    /* Encode operands based on field map */
    for (int i = 0; i < insn->num_operands && i < 4; i++) {
        const sass_operand_t *op = &insn->operands[i];
        int offset = -1;

        if (i == 0 && fm->dst_offset >= 0) offset = fm->dst_offset;
        else if (i == 1 && fm->src1_offset >= 0) offset = fm->src1_offset;
        else if (i == 2 && fm->src2_offset >= 0) offset = fm->src2_offset;
        else if (i == 3 && fm->src3_offset >= 0) offset = fm->src3_offset;

        if (offset >= 0) {
            if (op->type == SASS_REG_GPR || op->type == SASS_REG_PRED) {
                b[offset] = op->index & 0xFF;
                if (offset + 1 < 16) b[offset + 1] = 0x00;
            } else if (op->has_imm) {
                write_le32(b, offset, op->imm_value & 0xFFFFFFFF);
            }
        }
    }

    /* LUT for LOP3 - the LUT immediate is the 5th operand (index 4) */
    if (fm->lut_offset >= 0 && insn->num_operands > 4) {
        const sass_operand_t *op = &insn->operands[4];
        if (op->has_imm) {
            b[fm->lut_offset] = op->imm_value & 0xFF;
        }
    }

    /* Shift amount for SHF */
    if (fm->shift_offset >= 0 && insn->num_operands > 1) {
        const sass_operand_t *op = &insn->operands[1];
        if (op->has_imm) {
            b[fm->shift_offset] = op->imm_value & 0xFF;
        }
    }

    /* Flags byte */
    if (fm->flags_offset >= 0) {
        uint8_t flags = 0;
        /* Add .reuse flag if any operand has it */
        for (int i = 0; i < insn->num_operands; i++) {
            if (insn->operands[i].is_reuse) {
                flags |= 0x80;  /* reuse bit (empirical) */
                break;
            }
        }
        b[fm->flags_offset] = flags;
    }

    /* Control word at offset 12 */
    if (insn->has_predicate) {
        write_le32(b, 12, sass_ctrl_predicated(0, 0, 0, 0, 0, 
                                                 insn->predicate_reg, 
                                                 insn->predicate_negated));
    } else {
        write_le32(b, 12, sass_ctrl_plain());
    }

    return SASS_OK;
}

int sass_encode(sass_context_t *ctx, const sass_instruction_t *insn, sass_encoded_t *out) {
    if (!ctx || !insn || !out) return SASS_ERR_ENCODE;

    /* Special handling for MOV */
    if (strcasecmp(insn->mnemonic, "MOV") == 0) {
        if (insn->num_operands >= 2) {
            const sass_operand_t *src = &insn->operands[1];
            if (src->cbank >= 0) {
                /* c[bank][off] format */
                return encode_mov_const(ctx, insn, out);
            }
            return encode_mov_imm(ctx, insn, out);
        }
    }

    /* Special handling for branch instructions with backtick targets */
    if (strcasecmp(insn->mnemonic, "BRA") == 0 ||
        strcasecmp(insn->mnemonic, "BSSY") == 0 ||
        strcasecmp(insn->mnemonic, "BRX") == 0 ||
        strcasecmp(insn->mnemonic, "BAR.SYNC") == 0) {
        return encode_branch(ctx, insn, out);
    }

    /* Find field map */
    const sass_field_map_t *fm = find_field_map(ctx, insn->mnemonic);
    if (!fm) {
        /* Try with common suffixes */
        char alt_name[64];
        snprintf(alt_name, sizeof(alt_name), "%s.U32", insn->mnemonic);
        fm = find_field_map(ctx, alt_name);
        if (!fm) {
            snprintf(alt_name, sizeof(alt_name), "%s.LUT", insn->mnemonic);
            fm = find_field_map(ctx, alt_name);
        }
        if (!fm) {
            snprintf(alt_name, sizeof(alt_name), "%s.L.U32", insn->mnemonic);
            fm = find_field_map(ctx, alt_name);
        }
        if (!fm) {
            snprintf(alt_name, sizeof(alt_name), "%s.R.U32", insn->mnemonic);
            fm = find_field_map(ctx, alt_name);
        }
    }

    if (!fm) {
        return SASS_ERR_UNSUPPORTED;
    }

    return encode_generic(ctx, insn, fm, out);
}

int sass_assemble(sass_context_t *ctx, const char *line, sass_encoded_t *out) {
    sass_instruction_t insn;
    int rc = sass_parse_line(ctx, line, &insn);
    if (rc != SASS_OK) return rc;
    return sass_encode(ctx, &insn, out);
}

void sass_print_hex(const sass_encoded_t *enc, const char *comment) {
    for (int i = 0; i < SASS_INSN_SIZE; i++) {
        printf("%02x", enc->bytes[i]);
    }
    if (comment) printf("  // %s", comment);
    printf("\n");
}

const char *sass_strerror(int err) {
    static char buf[64];
    if (err >= 0 && err < (int)(sizeof(error_strings) / sizeof(error_strings[0]))) {
        return error_strings[err];
    }
    snprintf(buf, sizeof(buf), "unknown error %d", err);
    return buf;
}

int sass_init(sass_context_t *ctx, sass_arch_t arch) {
    if (!ctx) return SASS_ERR_ENCODE;
    memset(ctx, 0, sizeof(*ctx));
    ctx->arch = arch;
    ctx->conservative_ctrl = 1;

    switch (arch) {
        case SASS_ARCH_KEPLER_SM35:
        case SASS_ARCH_KEPLER_SM37:
            ctx->field_maps = kepler_field_maps;
            ctx->num_field_maps = kepler_num_field_maps;
            break;
        case SASS_ARCH_TURING_SM75:
            ctx->field_maps = turing_field_maps_extended;
            ctx->num_field_maps = turing_num_field_maps_extended;
            break;
        default:
            ctx->field_maps = turing_field_maps;
            ctx->num_field_maps = turing_num_field_maps;
            break;
    }
    return SASS_OK;
}

const sass_field_map_t *sass_get_field_maps(sass_arch_t arch, int *count) {
    switch (arch) {
        case SASS_ARCH_KEPLER_SM35:
        case SASS_ARCH_KEPLER_SM37:
            *count = kepler_num_field_maps;
            return kepler_field_maps;
        case SASS_ARCH_TURING_SM75:
            *count = turing_num_field_maps_extended;
            return turing_field_maps_extended;
        default:
            *count = turing_num_field_maps;
            return turing_field_maps;
    }
}