/*
 * sassasm — SASS assembler for NVIDIA GPUs (Kepler, Turing, etc.)
 * Codename: shyleaf
 * Status: UNRELEASED — internal development build
 *
 * Instruction parser
 */

#include "sassasm_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <strings.h>  /* strcasecmp, strncasecmp */

/* Error strings */
static const char *error_strings[] = {
    [0] = "success",
    [1] = "parse error",
    [2] = "encode error",
    [3] = "unsupported instruction",
    [4] = "register/value out of range",
    [5] = "invalid operand",
};

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

    /* Check for branch target: `(.L_x_123) or `3248 */
    if (tok[0] == '`') {
        op->type = SASS_REG_SPECIAL;
        op->index = 15;  /* branch target marker */
        op->is_branch_target = 1;
        op->is_reuse = has_reuse;
        
        /* Check if it's a numeric offset: `3248 */
        char *p = tok + 1;
        if (isdigit(*p) || (*p == '-' && isdigit(*(p+1)))) {
            op->has_imm = 1;
            op->imm_value = strtoll(p, NULL, 0);
        } else {
            /* Label reference: `(.L_x_123) - placeholder for now */
            op->has_imm = 1;
            op->imm_value = 0;
        }
        return SASS_OK;
    }

    /* Check for memory operand: [R1] or [R1+0x10] or [UR4+0x401] or [0x401] */
    if (tok[0] == '[') {
        op->type = SASS_REG_GPR;  /* base register (default) */
        const char *p = tok + 1;
        int base_reg = -1;
        int64_t offset = 0;
        int is_ur = 0;
        
        /* Parse base register: R or UR, or just an immediate offset */
        if (p[0] == 'R' || p[0] == 'r') {
            base_reg = atoi(p + 1);
            while (*p && *p != ']' && *p != '+' && *p != '-') p++;
        } else if (p[0] == 'U' || p[0] == 'u') {
            if (p[1] == 'R' || p[1] == 'r') {
                is_ur = 1;
                base_reg = atoi(p + 2);
                while (*p && *p != ']' && *p != '+' && *p != '-') p++;
            }
        } else if (isdigit(*p) || *p == '0' && (p[1] == 'x' || p[1] == 'X')) {
            /* Immediate address without base register: [0x401] */
            base_reg = 0;  /* Use R0 as dummy base */
            offset = strtoll(p, NULL, 0);
            while (*p && *p != ']') p++;
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
            op->is_ur_base = is_ur;
            op->is_reuse = has_reuse;
            if (is_ur) {
                op->type = SASS_REG_UNIFORM;
            }
            return SASS_OK;
        }
        return SASS_ERR_PARSE;
    }

    /* Check for c[bank][offset] or c[bank][reg+offset] */
    if (tolower(tok[0]) == 'c' && tok[1] == '[') {
        op->type = SASS_REG_GPR;  /* destination is GPR */
        int bank = 0, off = 0;
        int reg = -1;
        const char *p = tok + 2;
        bank = strtol(p, (char **)&p, 0);
        if (*p == ']') p++;
        if (*p == '[') p++;
        /* Check if next token is a register */
        if (*p == 'R' || *p == 'r') {
            reg = atoi(p + 1);
            while (*p && *p != ']' && *p != '+' && *p != '-') p++;
            if (*p == '+' || *p == '-') {
                int sign = (*p == '+') ? 1 : -1;
                p++;
                off = sign * strtoll(p, NULL, 0);
            }
        } else {
            off = strtol(p, NULL, 0);
        }
        op->cbank = bank;
        op->coffset = off;
        op->creg = reg;
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

    /* Check for negated predicate: !PT, !UPT, !UP0-!UP7 */
    if (tok[0] == '!') {
        int rc = sass_reg_from_string(tok + 1, &op->type, &op->index);
        if (rc == SASS_OK) {
            op->is_negated = 1;
            op->is_reuse = has_reuse;
            return SASS_OK;
        }
        return rc;
    }

    /* Check for bitwise NOT register: ~R9 */
    if (tok[0] == '~' && (tok[1] == 'R' || tok[1] == 'r')) {
        int rc = sass_reg_from_string(tok + 1, &op->type, &op->index);
        if (rc == SASS_OK) {
            op->is_not = 1;
            op->is_reuse = has_reuse;
            return SASS_OK;
        }
        return rc;
    }

    /* Check for negative register: -R2 */
    if (tok[0] == '-' && (tok[1] == 'R' || tok[1] == 'r')) {
        int rc = sass_reg_from_string(tok + 1, &op->type, &op->index);
        if (rc == SASS_OK) {
            op->is_negated = 1;
            op->is_reuse = has_reuse;
            return SASS_OK;
        }
        return rc;
    }

    /* Register */
    int rc = sass_reg_from_string(tok, &op->type, &op->index);
    if (rc == SASS_OK) {
        op->is_reuse = has_reuse;
    }
    return rc;
}

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
    if (strncasecmp(s, "UPT", 3) == 0) {
        *type = SASS_REG_PRED;  /* UPT uses same predicate register file */
        *index = atoi(s + 3);
        if (*index < 0 || *index > 7) return SASS_ERR_RANGE;
        return SASS_OK;
    }
    if (strcasecmp(s, "PT") == 0) {
        *type = SASS_REG_PRED;
        *index = 0;  /* PT is special predicate register */
        return SASS_OK;
    }
    if (strncasecmp(s, "UP", 2) == 0) {
        /* UP0-UP7 uniform predicates */
        *type = SASS_REG_PRED;
        *index = 8 + atoi(s + 2);  /* UP0-UP7 map to 8-15 */
        if (*index < 8 || *index > 15) return SASS_ERR_RANGE;
        return SASS_OK;
    }
    if (strncasecmp(s, "SR_", 3) == 0) {
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
        else if (strcasecmp(s + 3, "LTMASK") == 0) *index = 13;
        else *index = 14;  /* unknown SR */
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
            else if (index == 14) snprintf(buf, sizeof(buf), "SRZ");
            else if (index >= 16 && index <= 31) snprintf(buf, sizeof(buf), "B%d", index - 16);
            else snprintf(buf, sizeof(buf), "SPEC%d", index);
            break;
    }
    return buf;
}

int sass_parse_line(sass_context_t *ctx, const char *line, sass_instruction_t *insn) {
    (void)ctx;  /* unused for now */
    if (!line || !insn) return SASS_ERR_PARSE;
    memset(insn, 0, sizeof(*insn));

    char *trimmed = strdup_trimmed(line);
    if (!trimmed) return SASS_ERR_PARSE;

    /* Skip comments and empty lines */
    if (trimmed[0] == '\0' || trimmed[0] == '/' || trimmed[0] == '#' || trimmed[0] == ';') {
        free(trimmed);
        return SASS_ERR_PARSE;  /* caller should skip these */
    }

    /* Check for predication prefix: @P0/@!P0 or @UPT0/@!UPT0 or @UP0/@!UP0 */
    char *tokenize_start = trimmed;
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
                /* Store predication info in instruction */
                insn->has_predicate = 1;
                insn->predicate_reg = reg;
                insn->predicate_negated = negated;
                /* Find end of predicate */
                while (*p && !isspace(*p)) p++;
                /* Tokenization will start after the predicate */
            }
        } else if (strncasecmp(p, "UPT", 3) == 0) {
            /* Uniform predicate @UPT0/@!UPT0 */
            int reg = atoi(p + 3);
            if (reg >= 0 && reg <= 7) {
                insn->has_predicate = 1;
                insn->predicate_reg = reg;
                insn->predicate_negated = negated;
                while (*p && !isspace(*p)) p++;
            }
        } else if (strncasecmp(p, "UP", 2) == 0) {
            /* Uniform predicate @UP0/@!UP0 through @UP7/@!UP7 */
            int reg = atoi(p + 2);
            if (reg >= 0 && reg <= 7) {
                insn->has_predicate = 1;
                insn->predicate_reg = 8 + reg;  /* UP0-UP7 map to 8-15 */
                insn->predicate_negated = negated;
                while (*p && !isspace(*p)) p++;
            }
        }
        /* Skip whitespace after predicate */
        while (*p && isspace(*p)) p++;
        tokenize_start = p;
    } else {
        tokenize_start = trimmed;
    }

    /* Remove trailing semicolon */
    char *semi = strrchr(trimmed, ';');
    if (semi) *semi = '\0';

    /* Check for control code prefix: --:-:-:Y:0 */
    if (strncmp(trimmed, "--:", 3) == 0) {
        /* Parse control code - simplified for now */
        /* Skip to after control code */
        char *space = strchr(trimmed, ' ');
        if (space) {
            memmove(trimmed, space + 1, strlen(space));
        }
    }

    /* Tokenize: split by spaces and commas */
    char *tokens[16];
    int ntokens = 0;
    char *p = tokenize_start;
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