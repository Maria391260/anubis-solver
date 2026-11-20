#ifndef SASSASM_H
#define SASSASM_H

/*
 * sassasm — SASS assembler for NVIDIA GPUs (Kepler, Turing, etc.)
 * Codename: shyleaf
 * Status: UNRELEASED — internal development build
 */

#include <stdint.h>
#include <stddef.h>

/* SASS instruction is always 16 bytes */
#define SASS_INSN_SIZE 16

/* Architectures */
typedef enum {
    SASS_ARCH_KEPLER_SM35 = 35,
    SASS_ARCH_KEPLER_SM37 = 37,
    SASS_ARCH_TURING_SM75 = 75,
    SASS_ARCH_AMPERE_SM80 = 80,
    SASS_ARCH_HOPPER_SM90 = 90,
} sass_arch_t;

/* Register types */
typedef enum {
    SASS_REG_GPR = 0,      /* R0-R255 */
    SASS_REG_PRED = 1,     /* P0-P7 */
    SASS_REG_CCB = 2,      /* CC0-CC3 */
    SASS_REG_UNIFORM = 3,  /* UR0-UR... */
    SASS_REG_SPECIAL = 4,  /* RZ, SP, etc. */
} sass_reg_type_t;

/* Operand */
typedef struct {
    sass_reg_type_t type;
    int index;          /* register number */
    int is_negated;     /* for predicates */
    int is_reuse;       /* .reuse flag */
    int is_memory;      /* memory operand [R1+offset] */
    int is_ur_base;     /* memory base is UR register */
    int is_branch_target; /* branch target `(.L_x_123) */
    /* For constants/immediates */
    uint64_t imm_value;
    int has_imm;
    /* For c[bank][offset] */
    int cbank;
    int coffset;
} sass_operand_t;

/* Parsed instruction */
typedef struct {
    char mnemonic[64];
    int num_operands;
    sass_operand_t operands[8];
    int has_control_code;  /* explicit --:-:-:Y:0 style */
    uint32_t control_code; /* if has_control_code */
    /* Predication */
    int has_predicate;
    int predicate_reg;     /* 0-7 for P0-P7 */
    int predicate_negated; /* 1 if @!P0 */
} sass_instruction_t;

/* Encoded instruction (16 bytes) */
typedef struct {
    uint8_t bytes[SASS_INSN_SIZE];
} sass_encoded_t;

/* Field map for an instruction family */
typedef struct {
    const char *name;
    uint16_t opcode;      /* first 2 bytes (little-endian in file) */
    int dst_offset;
    int src1_offset;
    int src2_offset;
    int src3_offset;
    int flags_offset;
    int lut_offset;       /* for LOP3 */
    int shift_offset;     /* for SHF */
    const char *note;
} sass_field_map_t;

/* Error codes */
#define SASS_OK 0
#define SASS_ERR_PARSE -1
#define SASS_ERR_ENCODE -2
#define SASS_ERR_UNSUPPORTED -3
#define SASS_ERR_RANGE -4
#define SASS_ERR_INVALID_OP -5

/* Assembler context */
typedef struct {
    sass_arch_t arch;
    const sass_field_map_t *field_maps;
    int num_field_maps;
    /* For control word scheduling (future) */
    int conservative_ctrl;
} sass_context_t;

/* Initialize context for a given architecture */
int sass_init(sass_context_t *ctx, sass_arch_t arch);

/* Parse a single SASS instruction line */
int sass_parse_line(sass_context_t *ctx, const char *line, sass_instruction_t *insn);

/* Encode a parsed instruction to 16 bytes */
int sass_encode(sass_context_t *ctx, const sass_instruction_t *insn, sass_encoded_t *out);

/* Convenience: parse + encode in one call */
int sass_assemble(sass_context_t *ctx, const char *line, sass_encoded_t *out);

/* Print encoded instruction as hex */
void sass_print_hex(const sass_encoded_t *enc, const char *comment);

/* Control word helpers */
uint32_t sass_ctrl_plain(void);       /* dependency-free: 00e20f00 */
uint32_t sass_ctrl_const(void);       /* constant load: 00e40f00 */
uint32_t sass_ctrl_custom(int stall, int yield, int rbar, int wbar, int reuse);

/* Register helpers */
int sass_reg_from_string(const char *s, sass_reg_type_t *type, int *index);
const char *sass_reg_to_string(sass_reg_type_t type, int index);

/* Architecture-specific opcode tables */
const sass_field_map_t *sass_get_field_maps(sass_arch_t arch, int *count);

/* Error strings */
const char *sass_strerror(int err);

#endif /* SASSASM_H */