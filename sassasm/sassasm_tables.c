/*
 * sassasm — SASS assembler for NVIDIA GPUs (Kepler, Turing, etc.)
 * Codename: shyleaf
 * Status: UNRELEASED — internal development build
 *
 * Instruction field maps / opcode tables
 */

#include "sassasm_types.h"

/* ===================== Kepler (sm_35/sm_37) Field Maps ===================== */
/* Based on nvdisasm analysis of Kepler cubins. Opcode bytes are little-endian in file order. */
const sass_field_map_t kepler_field_maps[] = {
    /* MOV Rd, imm32 */
    {"LDS.U.128", 0x8379, 2, 4, -1, -1, -1, -1, -1,
      "LDS.U.128: opcode 79 83, dst+2, addr+4"},

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
    {"UPRMT", 0x2872, 2, 4, 6, 8, -1, -1, -1,
      "UPRMT: opcode 72 28, URdst+2, URsrc1+4, URsrc2+6, control+8"},

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
};

const int kepler_num_field_maps = sizeof(kepler_field_maps) / sizeof(kepler_field_maps[0]);


/* ===================== Turing (sm_75) Field Maps Extended ===================== */
/* These match the Python FIELD_MAPS plus additional instructions discovered */
const sass_field_map_t turing_field_maps_extended[] = {
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
    {"PLOP3.LUT", 0x1273, 2, 4, 6, 8, 10, 9, -1,
     "PLOP3: opcode 73 12, dst+2, src1+4, src2+6, src3+8, lut+9, flags+10"},
    {"STG.E", 0x8679, -1, -1, -1, -1, -1, -1, -1,
     "STG.E: address from uniform reg pair"},
    {"EXIT", 0x794d, -1, -1, -1, -1, -1, -1, -1, ""},
    {"NOP", 0x7918, -1, -1, -1, -1, -1, -1, -1, ""},

    /* Basic ALU instructions */
    {"IMAD", 0x1872, 2, 4, 6, -1, 10, -1, -1,
     "IMAD: opcode 72 18, dst+2, src1+4, src2+6, flags+10"},
    {"IMAD.WIDE.U32", 0x1a72, 2, 4, 6, -1, 10, -1, -1,
     "IMAD.WIDE: opcode 72 1a"},
    {"IMAD.U32", 0x1872, 2, 4, 6, -1, 10, -1, -1,
      "IMAD.U32: opcode 72 18, dst+2, src1+4, src2+6, flags+10"},
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

    /* S2R - special register to register */
    {"S2R", 0x2e72, 2, 4, -1, -1, -1, -1, -1,
     "S2R: opcode 72 2e, dst+2, src_special+4"},
    {"S2UR", 0x2f72, 2, 4, -1, -1, -1, -1, -1,
      "S2UR: opcode 72 2f, URdst+2, src_special+4"},
    {"R2UR", 0x2d72, 2, 4, -1, -1, -1, -1, -1,
      "R2UR: opcode 72 2d, URdst+2, GPRsrc+4"},
    {"CS2R", 0x2e72, 2, 4, -1, -1, -1, -1, -1,
      "CS2R: opcode 72 2e, dst+2, SR+4"},

    /* STL.128 - store local 128-bit */
    {"STL.128", 0x8579, -1, 4, -1, -1, -1, -1, -1,
     "STL.128: opcode 79 85, addr+4"},
{"STL.128", 0x8579, -1, 4, -1, -1, -1, -1, -1,
     "STL.128: opcode 79 85, addr+4"},
    {"STS.128", 0x8479, -1, 4, -1, -1, -1, -1, -1,
      "STS.128: opcode 79 84, addr+4"},
    {"LDS.U.128", 0x8379, 2, 4, -1, -1, -1, -1, -1,
      "LDS.U.128: opcode 79 83, dst+2, addr+4"},

    {"STL", 0x8779, -1, 4, -1, -1, -1, -1, -1,
     "STL: opcode 79 87, addr+4"},
    {"STL.U8", 0x8f79, -1, 4, -1, -1, -1, -1, -1,
     "STL.U8: opcode 79 8f, addr+4"},

    /* LDL.128 - load local 128-bit */
    {"LDL.128", 0x8379, 2, 4, -1, -1, -1, -1, -1,
     "LDL.128: opcode 79 83, dst+2, addr+4"},
    {"LDL", 0x8979, 2, 4, -1, -1, -1, -1, -1,
     "LDL: opcode 79 89, dst+2, addr+4"},
    {"LDL.U8", 0x9179, 2, 4, -1, -1, -1, -1, -1,
      "LDL.U8: opcode 79 91, dst+2, addr+4"},
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
{"VOTEU", 0x4e72, 2, 4, -1, -1, -1, -1, -1,
     "VOTEU: opcode 72 4e, dst+2, src+4"},
    {"VOTEU.ANY", 0x4e72, 2, 4, -1, -1, -1, -1, -1,
      "VOTEU.ANY: opcode 72 4e, dst+2, src+4"},

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
{"SHF.R.U32.HI", 0x1978, 2, -1, 8, -1, 9, -1, 4,
     "SHF.R.U32.HI: opcode 78 19, dst+2, shift+4, src+8, flags+9"},
    {"SHF.R.W.U32", 0x1978, 2, -1, 8, -1, 9, -1, 4,
      "SHF.R.W.U32: opcode 78 19, dst+2, shift+4, src+8, flags+9"},

    {"SHF.L.U32.HI", 0x1978, 2, -1, 8, -1, 9, -1, 4,
     "SHF.L.U32.HI: opcode 78 19, dst+2, shift+4, src+8, flags+9"},
    {"PRMT", 0x2872, 2, 4, 6, 8, -1, -1, -1,
     "PRMT: opcode 72 28, dst+2, src1+4, src2+6, control+8"},
{"PRMT", 0x2872, 2, 4, 6, 8, -1, -1, -1,
     "PRMT: opcode 72 28, dst+2, src1+4, src2+6, control+8"},
    {"UPRMT", 0x2872, 2, 4, 6, 8, -1, -1, -1,
      "UPRMT: opcode 72 28, URdst+2, URsrc1+4, URsrc2+6, control+8"},

    {"ATOMG.E.ADD.STRONG.GPU", 0x9079, -1, 4, 6, -1, -1, -1, -1,
     "ATOMG.STRONG: opcode 79 90, addr+4, src+6"},

    /* Uniform register instructions */
    {"ULDC.64", 0x2e72, 2, 4, -1, -1, -1, -1, -1,
     "ULDC.64: opcode 72 2e, URDst+2, c[bank][off]+4"},
    {"ULDC.U8", 0x2f72, 2, 4, -1, -1, -1, -1, -1,
      "ULDC.U8: opcode 72 2f, URDst+2, c[bank][off]+4"},
    {"ULDC", 0x2d72, 2, 4, -1, -1, -1, -1, -1,
      "ULDC: opcode 72 2d, URDst+2, c[bank][off]+4"},
    {"LDC.U8", 0x2f72, 2, 4, -1, -1, -1, -1, -1,
      "LDC.U8: opcode 72 2f, dst+2, c[bank][reg+off]+4"},
    {"LDC", 0x2d72, 2, 4, -1, -1, -1, -1, -1,
      "LDC: opcode 72 2d, dst+2, c[bank][reg+off]+4"},
{"LDC", 0x2d72, 2, 4, -1, -1, -1, -1, -1,
      "LDC: opcode 72 2d, dst+2, c[bank][reg+off]+4"},
    {"LDC.64", 0x2d72, 2, 4, -1, -1, -1, -1, -1,
      "LDC.64: opcode 72 2d, dst+2, c[bank][reg+off]+4"},

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
{"USHF.R.U64", 0x1b78, 2, -1, 8, -1, 9, -1, 4,
     "USHF.R.U64: opcode 78 1b, URDst+2, shift+4, URsrc+8, flags+9"},
    {"USHF.L.W.U32.HI", 0x1978, 2, -1, 8, -1, 9, -1, 4,
      "USHF.L.W.U32.HI: opcode 78 19, URDst+2, shift+4, URsrc+8, flags+9"},

    {"UIADD3", 0x1072, 2, 4, 6, 8, 10, -1, -1,
     "UIADD3: opcode 72 10, URDst+2, URsrc1+4, URsrc2+6, URsrc3+8, flags+10"},
    {"UIADD3.X", 0x1072, 2, 4, 6, 8, 10, -1, -1,
     "UIADD3.X: opcode 72 10, URDst+2, URsrc1+4, URsrc2+6, URsrc3+8, flags+10"},
    {"UISETP.GE.U32.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
      "UISETP.GE.U32.AND: opcode 72 1c, writes UP, URsrc1+4, URsrc2+6"},
    {"UISETP.GT.U32.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
      "UISETP.GT.U32.AND: opcode 72 1c, writes UP, URsrc1+4, URsrc2+6"},
{"UISETP.GT.U32.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
      "UISETP.GT.U32.AND: opcode 72 1c, writes UP, URsrc1+4, URsrc2+6"},
    {"UISETP.NE.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
      "UISETP.NE.AND: opcode 72 1c, writes UP, URsrc1+4, URsrc2+6"},
    {"UISETP.LT.U32.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
      "UISETP.LT.U32.AND: opcode 72 1c, writes UP, URsrc1+4, URsrc2+6"},
    {"UISETP.EQ.OR", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
      "UISETP.EQ.OR: opcode 72 1c, writes UP, URsrc1+4, URsrc2+6"},

    {"UIMAD.WIDE.U32", 0x1a72, 2, 4, 6, -1, 10, -1, -1,
      "UIMAD.WIDE.U32: opcode 72 1a, URDst+2, URsrc1+4, URsrc2+6, flags+10"},
{"UIMAD.WIDE.U32", 0x1a72, 2, 4, 6, -1, 10, -1, -1,
      "UIMAD.WIDE.U32: opcode 72 1a, URDst+2, URsrc1+4, URsrc2+6, flags+10"},
    {"UIMAD", 0x1872, 2, 4, 6, -1, 10, -1, -1,
      "UIMAD: opcode 72 18, URDst+2, URsrc1+4, URsrc2+6, flags+10"},

    {"ULEA", 0x6472, 2, 4, 6, 8, -1, -1, -1,
      "ULEA: opcode 72 64, URDst+2, URbase+4, URindex+6, offset+8"},
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
    {"LDG.E.U8.SYS", 0x9579, 2, 4, -1, -1, -1, -1, -1,
      "LDG.E.U8.SYS: opcode 79 95, dst+2, addr+4"},
    {"STG.E.64.SYS", 0x9479, 4, 6, -1, -1, -1, -1, -1,
     "STG.E.64.SYS: opcode 79 94, addr at dst+4, src at src1+6"},
{"STG.E.64.SYS", 0x9479, 4, 6, -1, -1, -1, -1, -1,
     "STG.E.64.SYS: opcode 79 94, addr at dst+4, src at src1+6"},
{"STG.E.64.SYS", 0x9479, 4, 6, -1, -1, -1, -1, -1,
     "STG.E.64.SYS: opcode 79 94, addr at dst+4, src at src1+6"},
{"STG.E.64.SYS", 0x9479, 4, 6, -1, -1, -1, -1, -1,
     "STG.E.64.SYS: opcode 79 94, addr at dst+4, src at src1+6"},
{"STG.E.64.SYS", 0x9479, 4, 6, -1, -1, -1, -1, -1,
     "STG.E.64.SYS: opcode 79 94, addr at dst+4, src at src1+6"},
{"STG.E.64.SYS", 0x9479, 4, 6, -1, -1, -1, -1, -1,
     "STG.E.64.SYS: opcode 79 94, addr at dst+4, src at src1+6"},
{"STG.E.64.SYS", 0x9479, 4, 6, -1, -1, -1, -1, -1,
     "STG.E.64.SYS: opcode 79 94, addr at dst+4, src at src1+6"},
{"STG.E.64.SYS", 0x9479, 4, 6, -1, -1, -1, -1, -1,
     "STG.E.64.SYS: opcode 79 94, addr at dst+4, src at src1+6"},
    {"STG.E.128.SYS", 0x9879, 4, 6, -1, -1, -1, -1, -1,
      "STG.E.128.SYS: opcode 79 98, addr at dst+4, src at src1+6"},

    {"STG.E.128.SYS", 0x9879, 4, 6, -1, -1, -1, -1, -1,
      "STG.E.128.SYS: opcode 79 98, addr at dst+4, src at src1+6"},

    {"STG.E.SYS", 0x9679, 4, 6, -1, -1, -1, -1, -1,
      "STG.E.SYS: opcode 79 96, addr at dst+4, src at src1+6"},
    {"LDG.E.SYS", 0x9779, 2, 4, -1, -1, -1, -1, -1,
      "LDG.E.SYS: opcode 79 97, dst+2, addr+4"},
{"LDG.E.SYS", 0x9779, 2, 4, -1, -1, -1, -1, -1,
      "LDG.E.SYS: opcode 79 97, dst+2, addr+4"},
    {"LDG.E.128.SYS", 0x8179, 2, 4, -1, -1, -1, -1, -1,
      "LDG.E.128.SYS: opcode 79 81, dst+2, addr+4"},

    {"STL.S16", 0x8879, -1, 4, -1, -1, -1, -1, -1,
      "STL.S16: opcode 79 88, addr+4"},
    {"STL.64", 0x8c79, -1, 4, -1, -1, -1, -1, -1,
      "STL.64: opcode 79 8c, addr+4"},

    {"LDS.U.U8", 0x8e79, 2, 4, -1, -1, -1, -1, -1,
     "LDS.U.U8: opcode 79 8e, dst+2, addr+4"},

    /* More LDS/STS variants */
    {"LDS.64", 0x8c79, 2, 4, -1, -1, -1, -1, -1,
     "LDS.64: opcode 79 8c, dst+2, addr+4"},
    {"STS.64", 0x8e79, -1, 4, -1, -1, -1, -1, -1,
     "STS.64: opcode 79 8e, addr+4"},
    {"LDS.U.64", 0x8e79, 2, 4, -1, -1, -1, -1, -1,
     "LDS.U.64: opcode 79 8e, dst+2, addr+4"},
    {"LDL.64", 0x8d79, 2, 4, -1, -1, -1, -1, -1,
      "LDL.64: opcode 79 8d, dst+2, addr+4"},
    {"STS.U8", 0x8f79, -1, 4, -1, -1, -1, -1, -1,
      "STS.U8: opcode 79 8f, addr+4"},
    {"STS.U.64", 0x9079, -1, 4, -1, -1, -1, -1, -1,
     "STS.U.64: opcode 79 90, addr+4"},

    /* FLO variants */
    {"FLO.U32", 0x5472, 2, 4, -1, -1, -1, -1, -1,
     "FLO.U32: opcode 72 54, dst+2, src+4"},

    /* More ISETP variants */
    {"ISETP.GT.U32.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
     "ISETP.GT: opcode 72 1c, writes predicate, src1+4, src2+6"},
    {"ISETP.GT.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
      "ISETP.GT.AND: opcode 72 1c, writes predicate, src1+4, src2+6"},
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
{"ISETP.NE.AND.EX", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
     "ISETP.NE.EX: opcode 72 1c, writes predicate, src1+4, src2+6"},
    {"ISETP.NE.U32.OR", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
      "ISETP.NE.U32.OR: opcode 72 1c, writes predicate, src1+4, src2+6"},
    {"ISETP.NE.OR", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
      "ISETP.NE.OR: opcode 72 1c, writes predicate, src1+4, src2+6"},

    {"ISETP.GE.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
      "ISETP.GE.AND: opcode 72 1c, writes predicate, src1+4, src2+6"},
    {"ISETP.LT.OR", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
      "ISETP.LT.OR: opcode 72 1c, writes predicate, src1+4, src2+6"},

    /* IMNMX variants */
{"ISETP.LT.OR", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
      "ISETP.LT.OR: opcode 72 1c, writes predicate, src1+4, src2+6"},

    /* IMNMX variants */
    {"ISETP.EQ.OR", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
      "ISETP.EQ.OR: opcode 72 1c, writes predicate, src1+4, src2+6"},

{"ISETP.EQ.OR", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
      "ISETP.EQ.OR: opcode 72 1c, writes predicate, src1+4, src2+6"},

    {"ISETP.EQ.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
      "ISETP.EQ.AND: opcode 72 1c, writes predicate, src1+4, src2+6"},

    {"IMNMX.U32", 0x4472, 2, 4, 6, -1, -1, -1, -1,
     "IMNMX.U32: opcode 72 44, dst+2, src1+4, src2+6"},
    {"IMNMX.S32", 0x4672, 2, 4, 6, -1, -1, -1, -1,
     "IMNMX.S32: opcode 72 46, dst+2, src1+4, src2+6"},

    /* LEA variants */
    {"LEA.HI", 0x6072, 2, 4, 6, 8, -1, -1, -1,
     "LEA.HI: opcode 72 60, dst+2, base+4, index+6, scale+8"},
    {"LEA.LO", 0x6272, 2, 4, 6, 8, -1, -1, -1,
     "LEA.LO: opcode 72 62, dst+2, base+4, index+6, scale+8"},
    {"LEA", 0x6472, 2, 4, 6, 8, -1, -1, -1,
      "LEA: opcode 72 64, dst+2, base+4, index+6, offset+8"},
    {"LEA.HI.X", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "LEA.HI.X: opcode 72 60, dst+2, base+4, index+6, offset+8, pred+10"},
    {"LEA.HI.X.SX32", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "LEA.HI.X.SX32: opcode 72 60, dst+2, base+4, index+6, offset+8, pred+10"},
    {"ULEA.HI.X", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "ULEA.HI.X: opcode 72 60, URdst+2, URbase+4, URindex+6, offset+8, UP+10"},

    /* BAR/MEMBAR */
    {"BAR.SYNC", 0x7a79, -1, -1, -1, -1, -1, -1, -1,
     "BAR.SYNC: opcode 79 7a"},
{"BAR.SYNC", 0x7a79, -1, -1, -1, -1, -1, -1, -1,
     "BAR.SYNC: opcode 79 7a"},
    {"BREAK", 0x7b79, -1, -1, -1, -1, -1, -1, -1,
      "BREAK: opcode 79 7b, barrier operand"},

    {"MEMBAR.SYS", 0x7b79, -1, -1, -1, -1, -1, -1, -1,
     "MEMBAR.SYS: opcode 79 7b"},
    {"MEMBAR.CTA", 0x7c79, -1, -1, -1, -1, -1, -1, -1,
     "MEMBAR.CTA: opcode 79 7c"},

    /* SEL variants */
{"MEMBAR.CTA", 0x7c79, -1, -1, -1, -1, -1, -1, -1,
     "MEMBAR.CTA: opcode 79 7c"},

    /* SEL variants */
    {"WARPSYNC", 0x7d79, -1, -1, -1, -1, -1, -1, -1,
      "WARPSYNC: opcode 79 7d, mask in immediate"},

    {"SEL", 0x2472, 2, 4, 6, 8, -1, -1, -1,
     "SEL: opcode 72 24, dst+2, cond+4, true+6, false+8"},
{"SEL", 0x2472, 2, 4, 6, 8, -1, -1, -1,
     "SEL: opcode 72 24, dst+2, cond+4, true+6, false+8"},
    {"USEL", 0x2472, 2, 4, 6, 8, -1, -1, -1,
      "USEL: opcode 72 24, URdst+2, cond+4, true+6, false+8"},

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

    /* Branch instructions */
    {"BRA", 0x8079, -1, -1, -1, -1, -1, -1, -1,
     "BRA: opcode 79 80, target in upper bytes"},
    {"BRX", 0x8179, -1, -1, -1, -1, -1, -1, -1,
     "BRX: opcode 79 81, target in upper bytes"},
    {"BSSY", 0x7879, -1, -1, -1, -1, -1, -1, -1,
     "BSSY: opcode 79 78, target in upper bytes"},
    {"BAR.SYNC", 0x7a79, -1, -1, -1, -1, -1, -1, -1,
     "BAR.SYNC: opcode 79 7a"},

    /* More barrier/sync */
    {"BSYNC", 0x7979, -1, -1, -1, -1, -1, -1, -1,
     "BSYNC: opcode 79 79, target in upper bytes"},
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

    /* FLO variants */
    {"FLO.U32", 0x5472, 2, 4, -1, -1, -1, -1, -1,
     "FLO.U32: opcode 72 54, dst+2, src+4"},

    /* SHF variants */
    {"SHF.R.S32.HI", 0x1b78, 2, -1, 8, -1, 9, -1, 4,
     "SHF.R.S32.HI: opcode 78 1b, dst+2, shift+4, src+8, flags+9"},
    {"SHFL.IDX", 0x5a72, 2, 4, 6, 8, -1, -1, -1,
     "SHFL.IDX: opcode 72 5a, dst+2, src+4, lane+6, width+8"},
    {"SHF.L.U64.HI", 0x1b78, 2, -1, 8, -1, 9, -1, 4,
     "SHF.L.U64.HI: opcode 78 1b, dst+2, shift+4, src+8, flags+9"},
    {"SHF.L.W.U32.HI", 0x1d78, 2, -1, 8, -1, 9, -1, 4,
     "SHF.L.W.U32.HI: opcode 78 1d, dst+2, shift+4, src+8, flags+9"},

    /* IMAD variants */
    {"IMAD.IADD", 0x1672, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.IADD: opcode 72 16, dst+2, src1+4, src2+6"},
    {"IMAD.SHL.U32", 0x1e72, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.SHL.U32: opcode 72 1e, dst+2, src1+4, src2+6, shift in src2"},
    {"IMAD.SHL.U32", 0x1a72, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.SHL.U32 (alt): opcode 72 1a, dst+2, src1+4, src2+6, shift in src2"},

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
    {"LEA", 0x6472, 2, 4, 6, 8, -1, -1, -1,
      "LEA: opcode 72 64, dst+2, base+4, index+6, offset+8"},
    {"LEA.HI.X", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "LEA.HI.X: opcode 72 60, dst+2, base+4, index+6, offset+8, pred+10"},
    {"LEA.HI.X.SX32", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "LEA.HI.X.SX32: opcode 72 60, dst+2, base+4, index+6, offset+8, pred+10"},
    {"ULEA.HI.X", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "ULEA.HI.X: opcode 72 60, URdst+2, URbase+4, URindex+6, offset+8, UP+10"},

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

    /* FLO variants */
    {"FLO.U32", 0x5472, 2, 4, -1, -1, -1, -1, -1,
     "FLO.U32: opcode 72 54, dst+2, src+4"},

    /* FCHK (float check) */
    {"FCHK", 0x5c72, 2, 4, -1, -1, -1, -1, -1,
     "FCHK: opcode 72 5c, dst+2, src+4"},

    /* Branch instructions */
    {"BRA", 0x8079, -1, -1, -1, -1, -1, -1, -1,
     "BRA: opcode 79 80, target in upper bytes"},
    {"BRX", 0x8179, -1, -1, -1, -1, -1, -1, -1,
     "BRX: opcode 79 81, target in upper bytes"},
    {"BSSY", 0x7879, -1, -1, -1, -1, -1, -1, -1,
     "BSSY: opcode 79 78, target in upper bytes"},
    {"BAR.SYNC", 0x7a79, -1, -1, -1, -1, -1, -1, -1,
     "BAR.SYNC: opcode 79 7a"},

    /* More barrier/sync */
    {"BSYNC", 0x7979, -1, -1, -1, -1, -1, -1, -1,
     "BSYNC: opcode 79 79, target in upper bytes"},
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

    /* FLO variants */
    {"FLO.U32", 0x5472, 2, 4, -1, -1, -1, -1, -1,
     "FLO.U32: opcode 72 54, dst+2, src+4"},

    /* FCHK (float check) */
    {"FCHK", 0x5c72, 2, 4, -1, -1, -1, -1, -1,
     "FCHK: opcode 72 5c, dst+2, src+4"},

    /* Branch instructions */
    {"BRA", 0x8079, -1, -1, -1, -1, -1, -1, -1,
     "BRA: opcode 79 80, target in upper bytes"},
    {"BRX", 0x8179, -1, -1, -1, -1, -1, -1, -1,
     "BRX: opcode 79 81, target in upper bytes"},
    {"BSSY", 0x7879, -1, -1, -1, -1, -1, -1, -1,
     "BSSY: opcode 79 78, target in upper bytes"},
    {"BAR.SYNC", 0x7a79, -1, -1, -1, -1, -1, -1, -1,
     "BAR.SYNC: opcode 79 7a"},

    /* More barrier/sync */
    {"BSYNC", 0x7979, -1, -1, -1, -1, -1, -1, -1,
     "BSYNC: opcode 79 79, target in upper bytes"},
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

    /* FLO variants */
    {"FLO.U32", 0x5472, 2, 4, -1, -1, -1, -1, -1,
     "FLO.U32: opcode 72 54, dst+2, src+4"},

    /* FCHK (float check) */
    {"FCHK", 0x5c72, 2, 4, -1, -1, -1, -1, -1,
     "FCHK: opcode 72 5c, dst+2, src+4"},

    /* Branch instructions */
    {"BRA", 0x8079, -1, -1, -1, -1, -1, -1, -1,
     "BRA: opcode 79 80, target in upper bytes"},
    {"BRX", 0x8179, -1, -1, -1, -1, -1, -1, -1,
     "BRX: opcode 79 81, target in upper bytes"},
    {"BSSY", 0x7879, -1, -1, -1, -1, -1, -1, -1,
     "BSSY: opcode 79 78, target in upper bytes"},
    {"BAR.SYNC", 0x7a79, -1, -1, -1, -1, -1, -1, -1,
     "BAR.SYNC: opcode 79 7a"},

    /* More barrier/sync */
    {"BSYNC", 0x7979, -1, -1, -1, -1, -1, -1, -1,
     "BSYNC: opcode 79 79, target in upper bytes"},
    {"MEMBAR.SYS", 0x7b79, -1, -1, -1, -1, -1, -1, -1,
     "MEMBAR.SYS: opcode 79 7b"},
    {"MEMBAR.CTA", 0x7c79, -1, -1, -1, -1, -1, -1, -1,
     "MEMBAR.CTA: opcode 79 7c"},

    /* 64-bit sys memory */
    {"LDG.E.64.SYS", 0x9379, 2, 4, -1, -1, -1, -1, -1,
     "LDG.E.64.SYS: opcode 79 93, dst+2, addr+4"},
    {"LDG.E.U8.SYS", 0x9579, 2, 4, -1, -1, -1, -1, -1,
      "LDG.E.U8.SYS: opcode 79 95, dst+2, addr+4"},
    {"STG.E.64.SYS", 0x9479, 4, 6, -1, -1, -1, -1, -1,
     "STG.E.64.SYS: opcode 79 94, addr at dst+4, src at src1+6"},
    {"LDS.U.U8", 0x8e79, 2, 4, -1, -1, -1, -1, -1,
     "LDS.U.U8: opcode 79 8e, dst+2, addr+4"},

    /* More LDS/STS variants */
    {"LDS.64", 0x8c79, 2, 4, -1, -1, -1, -1, -1,
     "LDS.64: opcode 79 8c, dst+2, addr+4"},
    {"STS.64", 0x8e79, -1, 4, -1, -1, -1, -1, -1,
     "STS.64: opcode 79 8e, addr+4"},
    {"LDS.U.64", 0x8e79, 2, 4, -1, -1, -1, -1, -1,
     "LDS.U.64: opcode 79 8e, dst+2, addr+4"},
    {"LDL.64", 0x8d79, 2, 4, -1, -1, -1, -1, -1,
      "LDL.64: opcode 79 8d, dst+2, addr+4"},
    {"STS.U8", 0x8f79, -1, 4, -1, -1, -1, -1, -1,
      "STS.U8: opcode 79 8f, addr+4"},
    {"STS.U.64", 0x9079, -1, 4, -1, -1, -1, -1, -1,
     "STS.U.64: opcode 79 90, addr+4"},

    /* IMAD variants */
    {"IMAD.IADD", 0x1672, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.IADD: opcode 72 16, dst+2, src1+4, src2+6"},
    {"IMAD.SHL.U32", 0x1e72, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.SHL.U32: opcode 72 1e, dst+2, src1+4, src2+6, shift in src2"},
    {"IMAD.SHL.U32", 0x1a72, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.SHL.U32 (alt): opcode 72 1a, dst+2, src1+4, src2+6, shift in src2"},

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
    {"LEA", 0x6472, 2, 4, 6, 8, -1, -1, -1,
      "LEA: opcode 72 64, dst+2, base+4, index+6, offset+8"},
    {"LEA.HI.X", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "LEA.HI.X: opcode 72 60, dst+2, base+4, index+6, offset+8, pred+10"},
    {"LEA.HI.X.SX32", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "LEA.HI.X.SX32: opcode 72 60, dst+2, base+4, index+6, offset+8, pred+10"},
    {"ULEA.HI.X", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "ULEA.HI.X: opcode 72 60, URdst+2, URbase+4, URindex+6, offset+8, UP+10"},

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

    /* FLO variants */
    {"FLO.U32", 0x5472, 2, 4, -1, -1, -1, -1, -1,
     "FLO.U32: opcode 72 54, dst+2, src+4"},

    /* FCHK (float check) */
    {"FCHK", 0x5c72, 2, 4, -1, -1, -1, -1, -1,
     "FCHK: opcode 72 5c, dst+2, src+4"},

    /* Branch instructions */
    {"BRA", 0x8079, -1, -1, -1, -1, -1, -1, -1,
     "BRA: opcode 79 80, target in upper bytes"},
    {"BRX", 0x8179, -1, -1, -1, -1, -1, -1, -1,
     "BRX: opcode 79 81, target in upper bytes"},
    {"BSSY", 0x7879, -1, -1, -1, -1, -1, -1, -1,
     "BSSY: opcode 79 78, target in upper bytes"},
    {"BAR.SYNC", 0x7a79, -1, -1, -1, -1, -1, -1, -1,
     "BAR.SYNC: opcode 79 7a"},

    /* More barrier/sync */
    {"BSYNC", 0x7979, -1, -1, -1, -1, -1, -1, -1,
     "BSYNC: opcode 79 79, target in upper bytes"},
    {"MEMBAR.SYS", 0x7b79, -1, -1, -1, -1, -1, -1, -1,
     "MEMBAR.SYS: opcode 79 7b"},
    {"MEMBAR.CTA", 0x7c79, -1, -1, -1, -1, -1, -1, -1,
     "MEMBAR.CTA: opcode 79 7c"},

    /* SHF variants */
    {"SHF.R.S32.HI", 0x1b78, 2, -1, 8, -1, 9, -1, 4,
     "SHF.R.S32.HI: opcode 78 1b, dst+2, shift+4, src+8, flags+9"},
    {"SHFL.IDX", 0x5a72, 2, 4, 6, 8, -1, -1, -1,
     "SHFL.IDX: opcode 72 5a, dst+2, src+4, lane+6, width+8"},
    {"SHF.L.U64.HI", 0x1b78, 2, -1, 8, -1, 9, -1, 4,
     "SHF.L.U64.HI: opcode 78 1b, dst+2, shift+4, src+8, flags+9"},
    {"SHF.L.W.U32.HI", 0x1d78, 2, -1, 8, -1, 9, -1, 4,
     "SHF.L.W.U32.HI: opcode 78 1d, dst+2, shift+4, src+8, flags+9"},

    /* IMAD variants */
    {"IMAD.IADD", 0x1672, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.IADD: opcode 72 16, dst+2, src1+4, src2+6"},
    {"IMAD.SHL.U32", 0x1e72, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.SHL.U32: opcode 72 1e, dst+2, src1+4, src2+6, shift in src2"},
    {"IMAD.SHL.U32", 0x1a72, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.SHL.U32 (alt): opcode 72 1a, dst+2, src1+4, src2+6, shift in src2"},

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
    {"LEA", 0x6472, 2, 4, 6, 8, -1, -1, -1,
      "LEA: opcode 72 64, dst+2, base+4, index+6, offset+8"},
    {"LEA.HI.X", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "LEA.HI.X: opcode 72 60, dst+2, base+4, index+6, offset+8, pred+10"},
    {"LEA.HI.X.SX32", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "LEA.HI.X.SX32: opcode 72 60, dst+2, base+4, index+6, offset+8, pred+10"},
    {"ULEA.HI.X", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "ULEA.HI.X: opcode 72 60, URdst+2, URbase+4, URindex+6, offset+8, UP+10"},

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

    /* FLO variants */
    {"FLO.U32", 0x5472, 2, 4, -1, -1, -1, -1, -1,
     "FLO.U32: opcode 72 54, dst+2, src+4"},

    /* FCHK (float check) */
    {"FCHK", 0x5c72, 2, 4, -1, -1, -1, -1, -1,
     "FCHK: opcode 72 5c, dst+2, src+4"},

    /* 64-bit sys memory */
    {"LDG.E.64.SYS", 0x9379, 2, 4, -1, -1, -1, -1, -1,
     "LDG.E.64.SYS: opcode 79 93, dst+2, addr+4"},
    {"LDG.E.U8.SYS", 0x9579, 2, 4, -1, -1, -1, -1, -1,
      "LDG.E.U8.SYS: opcode 79 95, dst+2, addr+4"},
    {"STG.E.64.SYS", 0x9479, 4, 6, -1, -1, -1, -1, -1,
     "STG.E.64.SYS: opcode 79 94, addr at dst+4, src at src1+6"},
    {"LDS.U.U8", 0x8e79, 2, 4, -1, -1, -1, -1, -1,
     "LDS.U.U8: opcode 79 8e, dst+2, addr+4"},

    /* More LDS/STS variants */
    {"LDS.64", 0x8c79, 2, 4, -1, -1, -1, -1, -1,
     "LDS.64: opcode 79 8c, dst+2, addr+4"},
    {"STS.64", 0x8e79, -1, 4, -1, -1, -1, -1, -1,
     "STS.64: opcode 79 8e, addr+4"},
    {"LDS.U.64", 0x8e79, 2, 4, -1, -1, -1, -1, -1,
     "LDS.U.64: opcode 79 8e, dst+2, addr+4"},
    {"LDL.64", 0x8d79, 2, 4, -1, -1, -1, -1, -1,
      "LDL.64: opcode 79 8d, dst+2, addr+4"},
    {"STS.U8", 0x8f79, -1, 4, -1, -1, -1, -1, -1,
      "STS.U8: opcode 79 8f, addr+4"},
    {"STS.U.64", 0x9079, -1, 4, -1, -1, -1, -1, -1,
     "STS.U.64: opcode 79 90, addr+4"},

    /* FLO variants */
    {"FLO.U32", 0x5472, 2, 4, -1, -1, -1, -1, -1,
     "FLO.U32: opcode 72 54, dst+2, src+4"},

    /* More ISETP variants */
    {"ISETP.GT.U32.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
     "ISETP.GT: opcode 72 1c, writes predicate, src1+4, src2+6"},
    {"ISETP.GT.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
      "ISETP.GT.AND: opcode 72 1c, writes predicate, src1+4, src2+6"},
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
    {"ISETP.GE.AND", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
      "ISETP.GE.AND: opcode 72 1c, writes predicate, src1+4, src2+6"},
    {"ISETP.LT.OR", 0x1c72, -1, 4, 6, -1, -1, -1, -1,
      "ISETP.LT.OR: opcode 72 1c, writes predicate, src1+4, src2+6"},

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
    {"LEA", 0x6472, 2, 4, 6, 8, -1, -1, -1,
      "LEA: opcode 72 64, dst+2, base+4, index+6, offset+8"},
    {"LEA.HI.X", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "LEA.HI.X: opcode 72 60, dst+2, base+4, index+6, offset+8, pred+10"},
    {"LEA.HI.X.SX32", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "LEA.HI.X.SX32: opcode 72 60, dst+2, base+4, index+6, offset+8, pred+10"},
    {"ULEA.HI.X", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "ULEA.HI.X: opcode 72 60, URdst+2, URbase+4, URindex+6, offset+8, UP+10"},

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

    /* FLO variants */
    {"FLO.U32", 0x5472, 2, 4, -1, -1, -1, -1, -1,
     "FLO.U32: opcode 72 54, dst+2, src+4"},

    /* FCHK (float check) */
    {"FCHK", 0x5c72, 2, 4, -1, -1, -1, -1, -1,
     "FCHK: opcode 72 5c, dst+2, src+4"},

    /* Branch instructions */
    {"BRA", 0x8079, -1, -1, -1, -1, -1, -1, -1,
     "BRA: opcode 79 80, target in upper bytes"},
    {"BRX", 0x8179, -1, -1, -1, -1, -1, -1, -1,
     "BRX: opcode 79 81, target in upper bytes"},
    {"BSSY", 0x7879, -1, -1, -1, -1, -1, -1, -1,
     "BSSY: opcode 79 78, target in upper bytes"},
    {"BAR.SYNC", 0x7a79, -1, -1, -1, -1, -1, -1, -1,
     "BAR.SYNC: opcode 79 7a"},

    /* More barrier/sync */
    {"BSYNC", 0x7979, -1, -1, -1, -1, -1, -1, -1,
     "BSYNC: opcode 79 79, target in upper bytes"},
    {"MEMBAR.SYS", 0x7b79, -1, -1, -1, -1, -1, -1, -1,
     "MEMBAR.SYS: opcode 79 7b"},
    {"MEMBAR.CTA", 0x7c79, -1, -1, -1, -1, -1, -1, -1,
     "MEMBAR.CTA: opcode 79 7c"},

    /* 64-bit sys memory */
    {"LDG.E.64.SYS", 0x9379, 2, 4, -1, -1, -1, -1, -1,
     "LDG.E.64.SYS: opcode 79 93, dst+2, addr+4"},
    {"LDG.E.U8.SYS", 0x9579, 2, 4, -1, -1, -1, -1, -1,
      "LDG.E.U8.SYS: opcode 79 95, dst+2, addr+4"},
    {"STG.E.64.SYS", 0x9479, 4, 6, -1, -1, -1, -1, -1,
     "STG.E.64.SYS: opcode 79 94, addr at dst+4, src at src1+6"},
    {"LDS.U.U8", 0x8e79, 2, 4, -1, -1, -1, -1, -1,
     "LDS.U.U8: opcode 79 8e, dst+2, addr+4"},

    /* More LDS/STS variants */
    {"LDS.64", 0x8c79, 2, 4, -1, -1, -1, -1, -1,
     "LDS.64: opcode 79 8c, dst+2, addr+4"},
    {"STS.64", 0x8e79, -1, 4, -1, -1, -1, -1, -1,
     "STS.64: opcode 79 8e, addr+4"},
    {"LDS.U.64", 0x8e79, 2, 4, -1, -1, -1, -1, -1,
     "LDS.U.64: opcode 79 8e, dst+2, addr+4"},
    {"LDL.64", 0x8d79, 2, 4, -1, -1, -1, -1, -1,
      "LDL.64: opcode 79 8d, dst+2, addr+4"},
    {"STS.U8", 0x8f79, -1, 4, -1, -1, -1, -1, -1,
      "STS.U8: opcode 79 8f, addr+4"},
    {"STS.U.64", 0x9079, -1, 4, -1, -1, -1, -1, -1,
     "STS.U.64: opcode 79 90, addr+4"},

    /* IMAD variants */
    {"IMAD.IADD", 0x1672, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.IADD: opcode 72 16, dst+2, src1+4, src2+6"},
    {"IMAD.SHL.U32", 0x1e72, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.SHL.U32: opcode 72 1e, dst+2, src1+4, src2+6, shift in src2"},
    {"IMAD.SHL.U32", 0x1a72, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.SHL.U32 (alt): opcode 72 1a, dst+2, src1+4, src2+6, shift in src2"},

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
    {"LEA", 0x6472, 2, 4, 6, 8, -1, -1, -1,
      "LEA: opcode 72 64, dst+2, base+4, index+6, offset+8"},
    {"LEA.HI.X", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "LEA.HI.X: opcode 72 60, dst+2, base+4, index+6, offset+8, pred+10"},
    {"LEA.HI.X.SX32", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "LEA.HI.X.SX32: opcode 72 60, dst+2, base+4, index+6, offset+8, pred+10"},
    {"ULEA.HI.X", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "ULEA.HI.X: opcode 72 60, URdst+2, URbase+4, URindex+6, offset+8, UP+10"},

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

    /* FLO variants */
    {"FLO.U32", 0x5472, 2, 4, -1, -1, -1, -1, -1,
     "FLO.U32: opcode 72 54, dst+2, src+4"},

    /* FCHK (float check) */
    {"FCHK", 0x5c72, 2, 4, -1, -1, -1, -1, -1,
     "FCHK: opcode 72 5c, dst+2, src+4"},

    /* 64-bit sys memory */
    {"LDG.E.64.SYS", 0x9379, 2, 4, -1, -1, -1, -1, -1,
     "LDG.E.64.SYS: opcode 79 93, dst+2, addr+4"},
    {"LDG.E.U8.SYS", 0x9579, 2, 4, -1, -1, -1, -1, -1,
      "LDG.E.U8.SYS: opcode 79 95, dst+2, addr+4"},
    {"STG.E.64.SYS", 0x9479, 4, 6, -1, -1, -1, -1, -1,
     "STG.E.64.SYS: opcode 79 94, addr at dst+4, src at src1+6"},
    {"LDS.U.U8", 0x8e79, 2, 4, -1, -1, -1, -1, -1,
     "LDS.U.U8: opcode 79 8e, dst+2, addr+4"},

    /* More LDS/STS variants */
    {"LDS.64", 0x8c79, 2, 4, -1, -1, -1, -1, -1,
     "LDS.64: opcode 79 8c, dst+2, addr+4"},
    {"STS.64", 0x8e79, -1, 4, -1, -1, -1, -1, -1,
     "STS.64: opcode 79 8e, addr+4"},
    {"LDS.U.64", 0x8e79, 2, 4, -1, -1, -1, -1, -1,
     "LDS.U.64: opcode 79 8e, dst+2, addr+4"},
    {"LDL.64", 0x8d79, 2, 4, -1, -1, -1, -1, -1,
      "LDL.64: opcode 79 8d, dst+2, addr+4"},
    {"STS.U8", 0x8f79, -1, 4, -1, -1, -1, -1, -1,
      "STS.U8: opcode 79 8f, addr+4"},
    {"STS.U.64", 0x9079, -1, 4, -1, -1, -1, -1, -1,
     "STS.U.64: opcode 79 90, addr+4"},

    /* IMAD variants */
    {"IMAD.IADD", 0x1672, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.IADD: opcode 72 16, dst+2, src1+4, src2+6"},
    {"IMAD.SHL.U32", 0x1e72, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.SHL.U32: opcode 72 1e, dst+2, src1+4, src2+6, shift in src2"},
    {"IMAD.SHL.U32", 0x1a72, 2, 4, 6, -1, -1, -1, -1,
     "IMAD.SHL.U32 (alt): opcode 72 1a, dst+2, src1+4, src2+6, shift in src2"},

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
    {"LEA", 0x6472, 2, 4, 6, 8, -1, -1, -1,
      "LEA: opcode 72 64, dst+2, base+4, index+6, offset+8"},
    {"LEA.HI.X", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "LEA.HI.X: opcode 72 60, dst+2, base+4, index+6, offset+8, pred+10"},
    {"LEA.HI.X.SX32", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "LEA.HI.X.SX32: opcode 72 60, dst+2, base+4, index+6, offset+8, pred+10"},
    {"ULEA.HI.X", 0x6072, 2, 4, 6, 8, 10, -1, -1,
      "ULEA.HI.X: opcode 72 60, URdst+2, URbase+4, URindex+6, offset+8, UP+10"},

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

    /* FLO variants */
    {"FLO.U32", 0x5472, 2, 4, -1, -1, -1, -1, -1,
     "FLO.U32: opcode 72 54, dst+2, src+4"},

    /* FCHK (float check) */
    {"FCHK", 0x5c72, 2, 4, -1, -1, -1, -1, -1,
     "FCHK: opcode 72 5c, dst+2, src+4"},
};

const int turing_num_field_maps_extended = sizeof(turing_field_maps_extended) / sizeof(turing_field_maps_extended[0]);


/* ===================== Turing (sm_75) Field Maps (Basic) ===================== */
/* These match the Python FIELD_MAPS */
const sass_field_map_t turing_field_maps[] = {
    {"MOV_IMM", 0x7802, 2, -1, -1, -1, -1, -1, -1,
     "MOV Rd, imm32"},
    {"MOV_CONST", 0x7a02, 2, -1, -1, -1, -1, -1, -1,
     "MOV Rd, c[b][o]"},
    {"IADD3", 0x1072, 2, 4, 6, 8, 10, -1, -1,
     "IADD3: 0x107a when src2 is constant"},
    {"LOP3.LUT", 0x1272, 2, 4, 6, 8, 10, 9, -1,
     "LOP3: LUT at +9, flags at +10"},
    {"PLOP3.LUT", 0x1273, 2, 4, 6, 8, 10, 9, -1,
     "PLOP3: opcode 73 12, dst+2, src1+4, src2+6, src3+8, lut+9, flags+10"},
    {"SHF.L.U32", 0x1978, 2, -1, 8, -1, 9, -1, 4,
     "SHF.L/R.U32: direction in flags"},
    {"STG.E", 0x8679, -1, -1, -1, -1, -1, -1, -1,
     "STG.E: address from uniform reg pair"},
    {"EXIT", 0x794d, -1, -1, -1, -1, -1, -1, -1, ""},
    {"NOP", 0x7918, -1, -1, -1, -1, -1, -1, -1, ""},
};
const int turing_num_field_maps = sizeof(turing_field_maps) / sizeof(turing_field_maps[0]);