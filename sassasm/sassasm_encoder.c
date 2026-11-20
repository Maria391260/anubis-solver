/*
 * sassasm — SASS assembler for NVIDIA GPUs (Kepler, Turing, etc.)
 * Codename: shyleaf
 * Status: UNRELEASED — internal development build
 *
 * Instruction encoder
 */

#include "sassasm_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <strings.h>  /* strcasecmp */

/* Forward declarations for tables defined in sassasm_tables.c */
extern const sass_field_map_t kepler_field_maps[];
extern int kepler_num_field_maps;
extern const sass_field_map_t turing_field_maps_extended[];
extern int turing_num_field_maps_extended;

/* Conservative control words */
static const uint8_t CTRL_PLAIN[4] = {0x00, 0xe2, 0x0f, 0x00};
static const uint8_t CTRL_CONST[4] = {0x00, 0xe4, 0x0f, 0x00};

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

/* Control word helpers */
uint32_t sass_ctrl_plain(void) {
    return *(uint32_t *)CTRL_PLAIN;
}

uint32_t sass_ctrl_const(void) {
    return *(uint32_t *)CTRL_CONST;
}

uint32_t sass_ctrl_custom(int stall, int yield, int rbar, int wbar, int reuse) {
    uint32_t ctrl = 0;
    ctrl |= (stall & 0xF);
    ctrl |= ((yield & 1) << 4);
    ctrl |= ((rbar & 1) << 5);
    ctrl |= ((wbar & 1) << 6);
    ctrl |= ((reuse & 1) << 7);
    ctrl |= 0x0F00;  /* byte 2 = 0x0f */
    return ctrl;
}

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

const sass_field_map_t *find_field_map(sass_context_t *ctx, const char *mnemonic) {
    for (int i = 0; i < ctx->num_field_maps; i++) {
        if (strcasecmp(ctx->field_maps[i].name, mnemonic) == 0) {
            return &ctx->field_maps[i];
        }
    }
    return NULL;
}

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

    /* Branch target offset in bytes (offset 4-11, 8 bytes for 64-bit offset) */
    for (int i = 0; i < insn->num_operands; i++) {
        const sass_operand_t *op = &insn->operands[i];
        if (op->is_branch_target && op->has_imm) {
            /* Encode branch offset as 64-bit signed integer at offset 4 */
            write_le32(b, 4, (uint32_t)(op->imm_value & 0xFFFFFFFF));
            write_le32(b, 8, (uint32_t)((op->imm_value >> 32) & 0xFFFFFFFF));
            break;
        }
    }

    /* Control word at offset 12 */
    if (ctx->conservative_ctrl) {
        write_le32(b, 12, sass_ctrl_plain());
    } else {
        write_le32(b, 12, sass_ctrl_plain());
    }

    return SASS_OK;
}

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
    if (src->has_imm) {
        write_le32(b, 12, sass_ctrl_plain());
    } else {
        write_le32(b, 12, sass_ctrl_const());
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
            if (op->is_memory) {
                /* Memory operand: [base+offset] */
                /* Encode base register at offset */
                b[offset] = op->index & 0xFF;
                if (offset + 1 < 16) {
                    if (op->is_ur_base) {
                        b[offset + 1] = 0x80;  /* UR base flag (empirical) */
                    } else {
                        b[offset + 1] = 0x00;
                    }
                }
                /* Encode offset at offset+4 (empirical) */
                if (offset + 4 < 16 && op->has_imm) {
                    write_le32(b, offset + 4, op->imm_value & 0xFFFFFFFF);
                }
            } else if (op->type == SASS_REG_GPR || op->type == SASS_REG_PRED) {
                b[offset] = op->index & 0xFF;
                if (offset + 1 < 16) {
                    if (op->is_negated) {
                        b[offset + 1] = 0x80;  /* negated register flag (empirical) */
                    } else {
                        b[offset + 1] = 0x00;
                    }
                }
            } else if (op->has_imm) {
                write_le32(b, offset, op->imm_value & 0xFFFFFFFF);
            }
        }
    }

    /* LUT for LOP3 */
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
    if (ctx->conservative_ctrl) {
        write_le32(b, 12, sass_ctrl_plain());
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
    static const char *error_strings[] = {
        [0] = "success",
        [1] = "parse error",
        [2] = "encode error",
        [3] = "unsupported instruction",
        [4] = "register/value out of range",
        [5] = "invalid operand",
    };
    static char buf[64];
    if (err >= 0 && err < (int)(sizeof(error_strings) / sizeof(error_strings[0]))) {
        return error_strings[-err];
    }
    snprintf(buf, sizeof(buf), "unknown error %d", err);
    return buf;
}

const sass_field_map_t *sass_get_field_maps(sass_arch_t arch, int *count) {
    extern const sass_field_map_t kepler_field_maps[];
    extern int kepler_num_field_maps;
    extern const sass_field_map_t turing_field_maps_extended[];
    extern int turing_num_field_maps_extended;
    
    switch (arch) {
        case SASS_ARCH_KEPLER_SM35:
        case SASS_ARCH_KEPLER_SM37:
            *count = kepler_num_field_maps;
            return kepler_field_maps;
        case SASS_ARCH_TURING_SM75:
            *count = turing_num_field_maps_extended;
            return turing_field_maps_extended;
        default:
            *count = turing_num_field_maps_extended;
            return turing_field_maps_extended;
    }
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
            ctx->field_maps = turing_field_maps_extended;
            ctx->num_field_maps = turing_num_field_maps_extended;
            break;
    }
    return SASS_OK;
}