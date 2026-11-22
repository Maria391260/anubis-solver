# Anubis on bare metal: a UEFI application that solves challenges over the
# network, with no OS underneath.
#
#   make                 # -> anubis.efi
#   make run             # boot it in QEMU with OVMF (needs ovmf + qemu)
#   make clean
#
# gnu-efi is the only build dependency (Debian: apt install gnu-efi).
ARCH      ?= x86_64
EFIINC    ?= /usr/include/efi
EFILIB    ?= /usr/lib
PREFIX    ?= /usr
LIBDIR    ?= $(EFILIB)
INSTALLDIR = $(PREFIX)/lib/gnuefi/apps

CFLAGS    += -I$(EFIINC) -I$(EFIINC)/$(ARCH) -Ishim/include -I. \
             -DGNU_EFI_USE_MS_ABI -fno-stack-protector -fpic -fshort-wchar \
             -mno-red-zone -O2 -Wall -Wextra -Wno-unused-parameter -std=gnu11
LDFLAGS   += -nostdlib -znocombreloc -T $(LIBDIR)/elf_$(ARCH)_efi.lds \
             -shared -Bsymbolic -L$(LIBDIR) $(LIBDIR)/crt0-efi-$(ARCH).o
LIBS      += -lefi -lgnuefi

OBJS      := crt.o cpu.o caps.o json.o sha256.o timer.o mp.o net.o solve.o main.o

all: anubis.efi

anubis.efi: anubis.so
	# .rodata must be listed: gnu-efi's sample list omits it, which produces
	# an image whose string literals are missing. Keeping the list explicit
	# (rather than dumping every section) is what the OVMF loader accepts.
	objcopy -j .text -j .rodata -j .sdata -j .data -j .dynamic -j .dynsym \
	        -j .rel -j .rela -j .reloc --target=efi-app-$(ARCH) $< $@

anubis.so: $(OBJS)
	ld $(LDFLAGS) $(OBJS) $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: anubis.efi
	qemu-system-x86_64 -M q35 -m 512 -nographic \
	    -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd \
	    -drive if=pflash,format=raw,file=/tmp/anubis-vars.fd \
	    -netdev user,id=n0,hostfwd=tcp:127.0.0.1:8642-:8642 \
	    -device virtio-net-pci,netdev=n0 \
	    -drive format=raw,file=fat:rw:esp

esp: anubis.efi
	mkdir -p esp/EFI/BOOT
	cp anubis.efi esp/EFI/BOOT/BOOTX64.EFI

clean:
	rm -f *.o *.so *.efi
	rm -rf esp

.PHONY: all run clean esp
