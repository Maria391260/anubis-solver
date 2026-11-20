# sassasm — SASS assembler in C
# Build: make
# Install: make install PREFIX=/usr/local

CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11 -fPIC
LDFLAGS ?= -shared
PREFIX  ?= /usr/local

TARGET  = sassasm
SRC     = sassasm_parser.c sassasm_encoder.c sassasm_tables.c main.c
OBJ     = $(SRC:.c=.o)
LIB     = libsassasm.so
INCLUDE = sassasm_types.h

all: $(TARGET) $(LIB)

$(TARGET): $(OBJ)
	$(CC) -o $@ $^

$(LIB): sassasm_parser.o sassasm_encoder.o sassasm_tables.o
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c $(INCLUDE)
	$(CC) $(CFLAGS) -I. -c $< -o $@

install: $(TARGET) $(LIB) $(INCLUDE)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -d $(DESTDIR)$(PREFIX)/lib
	install -d $(DESTDIR)$(PREFIX)/include
	install -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/bin/
	install -m 755 $(LIB) $(DESTDIR)$(PREFIX)/lib/
	install -m 644 $(INCLUDE) $(DESTDIR)$(PREFIX)/include/

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/$(TARGET)
	rm -f $(DESTDIR)$(PREFIX)/lib/$(LIB)
	rm -f $(DESTDIR)$(PREFIX)/include/$(INCLUDE)

test: $(TARGET)
	@echo "=== Testing MOV immediate ==="
	./$(TARGET) -a kepler 'MOV R0, 0x2a'
	./$(TARGET) -a kepler 'MOV R1, 0xdeadbeef'
	./$(TARGET) -a kepler 'MOV R2, 42'
	@echo
	@echo "=== Testing MOV constant ==="
	./$(TARGET) -a kepler 'MOV R0, c[0x0][0x28]'
	./$(TARGET) -a kepler 'MOV R1, c[0x1][0x100]'
	@echo
	@echo "=== Testing IADD3 ==="
	./$(TARGET) -a kepler 'IADD3 R2, R3, R4, R5'
	@echo
	@echo "=== Testing LOP3 ==="
	./$(TARGET) -a kepler 'LOP3.LUT R0, R1, R2, R3, 0xfc'
	@echo
	@echo "=== Testing SHF ==="
	./$(TARGET) -a kepler 'SHF.L.U32 R0, 0x8, R1'
	./$(TARGET) -a kepler 'SHF.R.U32 R0, 0x10, R1'
	@echo
	@echo "=== Testing other instructions ==="
	./$(TARGET) -a kepler 'EXIT'
	./$(TARGET) -a kepler 'NOP'
	./$(TARGET) -a kepler 'IMAD R0, R1, R2, R3'
	./$(TARGET) -a kepler 'FADD R0, R1, R2'
	./$(TARGET) -a kepler 'FFMA R0, R1, R2, R3'
	@echo
	@echo "=== Testing Turing (sm_75) ==="
	./$(TARGET) -a turing 'MOV R0, 0x2a'
	./$(TARGET) -a turing 'MOV R1, c[0x0][0x28]'
	./$(TARGET) -a turing 'IADD3 R2, R3, R4, R5'
	./$(TARGET) -a turing 'LOP3.LUT R0, R1, R2, R3, 0xfc'
	./$(TARGET) -a turing 'SHF.L.U32 R0, 0x8, R1'
	./$(TARGET) -a turing 'SHF.R.U32.HI R3, RZ, 0x10, R0.reuse'
	./$(TARGET) -a turing 'PRMT R3, R3, 0x7604, R0'
	./$(TARGET) -a turing 'IMAD.MOV.U32 R2, RZ, RZ, 0x4'
	./$(TARGET) -a turing 'UMOV UR4, URZ'
	./$(TARGET) -a turing 'BMOV.32.CLEAR RZ, B0'
	./$(TARGET) -a turing 'BSSY B0, `(.L_x_16)'
	./$(TARGET) -a turing 'EXIT'
	./$(TARGET) -a turing 'NOP'
	./$(TARGET) -a turing 'IMAD R0, R1, R2, R3'
	./$(TARGET) -a turing 'FADD R0, R1, R2'
	./$(TARGET) -a turing 'FFMA R0, R1, R2, R3'
	./$(TARGET) -a turing 'S2R R0, SR_CTAID.X'
	./$(TARGET) -a turing 'S2R R3, SR_TID.X'
	./$(TARGET) -a turing 'SHF.R.U64 R7, R4, 0x3, R5'
	./$(TARGET) -a turing 'SHF.L.U64.HI R10, R10, 0x10, R11'
	./$(TARGET) -a turing 'SHF.L.W.U32.HI R2, R10.reuse, 0x15, R3'
	./$(TARGET) -a turing 'ULDC.64 UR8, c[0x0][0x168]'
	./$(TARGET) -a turing 'ULOP3.LUT UR7, UR8, 0xee, URZ, 0x3c'
	./$(TARGET) -a turing 'UIADD3 UR7, UR7, UR10, URZ'
	./$(TARGET) -a turing 'UIADD3.X UR8, UR9, UR11, URZ'
	./$(TARGET) -a turing 'LDG.E.64.SYS R2, [R16+0x300430]'
	./$(TARGET) -a turing 'STG.E.64.SYS [R16+0x300430], R2'
	./$(TARGET) -a turing 'LDS.U.U8 R2, [R0+UR4+0x400]'
	./$(TARGET) -a turing 'ISETP.GE.U32.AND P0, PT, R0, c[0x0][0x198], PT'
	./$(TARGET) -a turing 'ISETP.LT.U32.AND P0, PT, R3, R2, PT'
	./$(TARGET) -a turing 'ISETP.EQ.U32.AND P2, PT, R3, R2, PT'
	./$(TARGET) -a turing 'ISETP.NE.AND.EX P0, PT, R3, RZ, PT, P0'
	./$(TARGET) -a turing 'IMNMX.U32 R8, R8, 0x2, PT'
	./$(TARGET) -a turing 'IMNMX.S32 R8, R8, 0x3, PT'
	./$(TARGET) -a turing 'LEA.HI R5, R4, R1, RZ, 0x1d'
	./$(TARGET) -a turing 'LEA.LO R5, R4, R1, RZ, 0x1d'
	./$(TARGET) -a turing 'BAR.SYNC 0x0'
	./$(TARGET) -a turing 'MEMBAR.SYS'
	./$(TARGET) -a turing 'MEMBAR.CTA'
	./$(TARGET) -a turing 'SEL R4, R5, 0x1054, R35'
	./$(TARGET) -a turing 'BFE.U32 R3, R3, 0x30, RZ, 0xfc'
	./$(TARGET) -a turing 'PSETP.AND P2, PT, R3, R2, PT'
	./$(TARGET) -a turing 'VOTE.ANY P0, PT, R7, R7'
	./$(TARGET) -a turing 'FCHK R31, R31'
	./$(TARGET) -a turing 'BRA `(.L_x_15)'
	./$(TARGET) -a turing 'BSSY B0, `(.L_x_16)'
	./$(TARGET) -a turing 'BAR.SYNC 0x0'
	./$(TARGET) -a turing '@P0 BRA `(.L_x_13)'
	./$(TARGET) -a turing '@!P0 EXIT'
	./$(TARGET) -a turing '@P0 IMAD R0, R1, R2, R3'
	./$(TARGET) -a turing '@!P1 IADD3 R2, R3, R4, R5'
	@echo
	@echo "=== List families ==="
	./$(TARGET) -a kepler -l

clean:
	rm -f $(OBJ) $(TARGET) $(LIB)

# HASS tools
HASS_TOOLS = sass2hass.py hass2sass.py convert_all.py

hass: $(TARGET) $(LIB)
	@echo "=== HASS Tools ==="
	@echo "sass2hass.py   - Convert SASS to HASS"
	@echo "hass2sass.py   - Convert HASS to SASS or binary"
	@echo "convert_all.py - Batch convert all kernels"

test-hass: $(TARGET)
	@echo "=== Testing SASS -> HASS -> Binary round-trip ==="
	python3 sass2hass.py ../src/nvidia/sass/nvidiaT4_sha256.sass -o /tmp/test.hass --offsets
	python3 hass2sass.py /tmp/test.hass -b /tmp/test.bin
	@echo "Round-trip test passed!"

test-all: $(TARGET)
	python3 convert_all.py

.PHONY: all install uninstall test clean hass test-hass test-all