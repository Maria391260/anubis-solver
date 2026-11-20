/*
 * sassasm — SASS assembler for NVIDIA GPUs (Kepler, Turing, etc.)
 * Codename: shyleaf
 * Status: UNRELEASED — internal development build
 *
 * Main CLI entry point
 */

#include "sassasm_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <strings.h>
#include <getopt.h>

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options] 'INSTRUCTION'\n"
        "\n"
        "NVIDIA SASS Assembler for Kepler (sm_35/37), Turing (sm_75), and later.\n"
        "\n"
        "Options:\n"
        "  -a, --arch ARCH     Architecture: kepler (sm_35), kepler37 (sm_37), turing (sm_75)\n"
        "  -f, --file FILE     Read instructions from file (one per line)\n"
        "  -o, --output FILE   Write binary output to file\n"
        "  -j, --json          Output JSON format\n"
        "  -l, --list          List supported instruction families\n"
        "  -v, --verbose       Verbose output\n"
        "  -h, --help          Show this help\n"
        "\n"
        "Examples:\n"
        "  %s 'MOV R0, 0x2a'\n"
        "  %s 'MOV R1, c[0x0][0x28]'\n"
        "  %s -a kepler 'IADD3 R2, R3, R4, R5'\n"
        "  %s -f kernel.sass -o kernel.bin\n"
        "\n"
        "Supported instruction families (Kepler):\n"
        "  MOV, IADD3, LOP3.LUT, SHF.L/R.U32, STG.E, EXIT, NOP,\n"
        "  IMAD, IMAD.WIDE.U32, ISETP, BRA, LDS.128, STS.128,\n"
        "  LDG.128, PRMT, BFE.U32, BFI, SEL, FADD, FMUL, FFMA,\n"
        "  MUFU, RRO, HMUL, HFMA\n",
        prog, prog, prog, prog, prog);
}

static void list_families(sass_arch_t arch) {
    int count;
    const sass_field_map_t *maps = sass_get_field_maps(arch, &count);
    printf("Instruction families for %s:\n",
           arch == SASS_ARCH_KEPLER_SM35 ? "Kepler (sm_35)" :
           arch == SASS_ARCH_KEPLER_SM37 ? "Kepler (sm_37)" : "Turing (sm_75)");
    for (int i = 0; i < count; i++) {
        printf("  %-16s  opcode: 0x%04x  %s\n",
               maps[i].name, maps[i].opcode, maps[i].note);
    }
}

static sass_arch_t parse_arch(const char *s) {
    if (strcasecmp(s, "kepler") == 0 || strcasecmp(s, "sm_35") == 0 || strcasecmp(s, "35") == 0)
        return SASS_ARCH_KEPLER_SM35;
    if (strcasecmp(s, "kepler37") == 0 || strcasecmp(s, "sm_37") == 0 || strcasecmp(s, "37") == 0)
        return SASS_ARCH_KEPLER_SM37;
    if (strcasecmp(s, "turing") == 0 || strcasecmp(s, "sm_75") == 0 || strcasecmp(s, "75") == 0)
        return SASS_ARCH_TURING_SM75;
    if (strcasecmp(s, "ampere") == 0 || strcasecmp(s, "sm_80") == 0 || strcasecmp(s, "80") == 0)
        return SASS_ARCH_AMPERE_SM80;
    if (strcasecmp(s, "hopper") == 0 || strcasecmp(s, "sm_90") == 0 || strcasecmp(s, "90") == 0)
        return SASS_ARCH_HOPPER_SM90;
    fprintf(stderr, "Unknown architecture: %s\n", s);
    exit(1);
}

int main(int argc, char **argv) {
    static struct option long_opts[] = {
        {"arch", required_argument, 0, 'a'},
        {"file", required_argument, 0, 'f'},
        {"output", required_argument, 0, 'o'},
        {"json", no_argument, 0, 'j'},
        {"list", no_argument, 0, 'l'},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    sass_arch_t arch = SASS_ARCH_KEPLER_SM35;
    const char *input_file = NULL;
    const char *output_file = NULL;
    int json_out = 0;
    int list_only = 0;

    int opt;
    while ((opt = getopt_long(argc, argv, "a:f:o:jlvh", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'a': arch = parse_arch(optarg); break;
            case 'f': input_file = optarg; break;
            case 'o': output_file = optarg; break;
            case 'j': json_out = 1; break;
            case 'l': list_only = 1; break;
            case 'v': break;
            case 'h': print_usage(argv[0]); return 0;
            default: print_usage(argv[0]); return 1;
        }
    }

    if (list_only) {
        list_families(arch);
        return 0;
    }

    sass_context_t ctx;
    if (sass_init(&ctx, arch) != SASS_OK) {
        fprintf(stderr, "Failed to initialize assembler\n");
        return 1;
    }

    FILE *out_fp = stdout;
    if (output_file) {
        out_fp = fopen(output_file, json_out ? "w" : "wb");
        if (!out_fp) {
            perror("fopen output");
            return 1;
        }
    }

    if (input_file) {
        /* Read from file */
        FILE *fp = fopen(input_file, "r");
        if (!fp) {
            perror("fopen input");
            return 1;
        }

        char line[1024];
        int line_num = 0;
        if (json_out) fprintf(out_fp, "[\n");

        while (fgets(line, sizeof(line), fp)) {
            line_num++;
            char *trimmed = line;
            while (*trimmed && isspace(*trimmed)) trimmed++;
            if (*trimmed == '\0' || *trimmed == '#' || *trimmed == '/' || *trimmed == ';')
                continue;

            sass_encoded_t enc;
            int rc = sass_assemble(&ctx, trimmed, &enc);
            if (rc != SASS_OK) {
                fprintf(stderr, "Line %d: %s: %s\n", line_num, sass_strerror(rc), trimmed);
                if (json_out) fprintf(out_fp, "  {\"line\": %d, \"error\": \"%s\", \"text\": \"%s\"},\n",
                                      line_num, sass_strerror(rc), trimmed);
                continue;
            }

            if (json_out) {
                fprintf(out_fp, "  {\"line\": %d, \"bytes\": \"", line_num);
                for (int i = 0; i < SASS_INSN_SIZE; i++) {
                    fprintf(out_fp, "%02x", enc.bytes[i]);
                }
                fprintf(out_fp, "\", \"text\": \"%s\"},\n", trimmed);
            } else if (output_file) {
                fwrite(enc.bytes, 1, SASS_INSN_SIZE, out_fp);
            } else {
                sass_print_hex(&enc, trimmed);
            }
        }

        if (json_out) fprintf(out_fp, "]\n");
        fclose(fp);
    } else if (optind < argc) {
        /* Single instruction from command line */
        char *line = argv[optind];
        for (int i = optind + 1; i < argc; i++) {
            /* Join remaining args with spaces */
            size_t len = strlen(line);
            line = realloc(line, len + strlen(argv[i]) + 2);
            line[len] = ' ';
            strcpy(line + len + 1, argv[i]);
        }

        sass_encoded_t enc;
        int rc = sass_assemble(&ctx, line, &enc);
        if (rc != SASS_OK) {
            fprintf(stderr, "%s: %s\n", sass_strerror(rc), line);
            return 1;
        }

        if (json_out) {
            fprintf(out_fp, "{\"bytes\": \"");
            for (int i = 0; i < SASS_INSN_SIZE; i++) {
                fprintf(out_fp, "%02x", enc.bytes[i]);
            }
            fprintf(out_fp, "\", \"text\": \"%s\"}\n", line);
        } else if (output_file) {
            fwrite(enc.bytes, 1, SASS_INSN_SIZE, out_fp);
        } else {
            sass_print_hex(&enc, line);
        }
    } else {
        print_usage(argv[0]);
        return 1;
    }

    if (out_fp != stdout) fclose(out_fp);
    return 0;
}