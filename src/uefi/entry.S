/* The PE entry point. No gnu-efi crt0: it does relocation/init work that
 * faults on this toolchain, and we only need three things from the firmware -
 * the image handle and the system table, which arrive in RCX/RDX (MS ABI) and
 * get moved into the SysV argument registers for the C side. */
	.text
	.globl _start
	.type _start, @function
_start:
	subq	$0x28, %rsp          /* shadow space + 16-byte alignment */
	movq	%rcx, %rdi           /* ImageHandle */
	movq	%rdx, %rsi           /* SystemTable */
	call	anubis_main
	addq	$0x28, %rsp
	ret
	.size _start, .-_start
