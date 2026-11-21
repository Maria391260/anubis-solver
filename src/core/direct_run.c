/*
 * direct_run.c — what `./libanubis_solver.so` does when someone runs it.
 *
 * A shared object is not a program. It has no main(), and normally the kernel
 * never executes it. But `./libanubis_solver.so` is a thing people type, and a
 * '-shared' link leaves the ELF entry point at 0: the kernel loads the ET_DYN
 * file as a static PIE, jumps to address 0 and the process dies with SIGSEGV.
 * That is a confusing first impression, so the Makefile points the entry point
 * at anubis_lib_entry(). The stub below says what the file is, who made it and
 * how to actually use it, then exits 1.
 *
 * The stub cannot use libc. The kernel enters it straight from execve() with no
 * dynamic loader, so no relocations have been applied, no libc start-up has run
 * and there is no TLS. Everything here is therefore a raw system call, and the
 * message is a function-local static array that the compiler addresses
 * PC-relative - no GOT, no PLT.
 *
 * dlopen() never looks at the ELF entry point, so loading the library still
 * works exactly as before.
 */

#include <stddef.h>

#if defined(__x86_64__)
#define ANUBIS_SYS_WRITE 1L
#define ANUBIS_SYS_EXIT_GROUP 231L

static long anubis_raw_write(int fd, const void *buf, size_t len)
{
    register long rax __asm__("rax") = ANUBIS_SYS_WRITE;
    register long rdi __asm__("rdi") = fd;
    register long rsi __asm__("rsi") = (long)(size_t)buf;
    register long rdx __asm__("rdx") = (long)len;

    __asm__ volatile ("syscall"
                      : "+r"(rax)
                      : "r"(rdi), "r"(rsi), "r"(rdx)
                      : "rcx", "r11", "memory");
    return rax;
}

static void anubis_raw_exit(int code) __attribute__((noreturn));
static void anubis_raw_exit(int code)
{
    register long rax __asm__("rax") = ANUBIS_SYS_EXIT_GROUP;
    register long rdi __asm__("rdi") = code;

    __asm__ volatile ("syscall"
                      : "+r"(rax)
                      : "r"(rdi)
                      : "rcx", "r11", "memory");
    __builtin_unreachable();
}
#elif defined(__aarch64__)
#define ANUBIS_SYS_WRITE 64L
#define ANUBIS_SYS_EXIT_GROUP 94L

static long anubis_raw_write(int fd, const void *buf, size_t len)
{
    register long x8 __asm__("x8") = ANUBIS_SYS_WRITE;
    register long x0 __asm__("x0") = fd;
    register long x1 __asm__("x1") = (long)(size_t)buf;
    register long x2 __asm__("x2") = (long)len;

    __asm__ volatile ("svc 0"
                      : "+r"(x0)
                      : "r"(x1), "r"(x2), "r"(x8)
                      : "memory");
    return x0;
}

static void anubis_raw_exit(int code) __attribute__((noreturn));
static void anubis_raw_exit(int code)
{
    register long x8 __asm__("x8") = ANUBIS_SYS_EXIT_GROUP;
    register long x0 __asm__("x0") = code;

    __asm__ volatile ("svc 0"
                      : "+r"(x0)
                      : "r"(x8)
                      : "memory");
    __builtin_unreachable();
}
#else
/* Unknown architecture: plain libc calls. Good enough on arches where the
 * kernel enters the stub with a loader, and never reached on x86-64/AArch64. */
extern long write(int, const void *, unsigned long);
extern void _exit(int) __attribute__((noreturn));

static long anubis_raw_write(int fd, const void *buf, size_t len)
{
    return write(fd, buf, (unsigned long)len);
}

static void anubis_raw_exit(int code)
{
    _exit(code);
}
#endif

static void anubis_write_all(int fd, const char *buf, size_t len)
{
    while (len > 0) {
        long n = anubis_raw_write(fd, buf, len);

        if (n <= 0) {
            return;
        }
        buf += (size_t)n;
        len -= (size_t)n;
    }
}

/*
 * The ELF entry point (Makefile: -Wl,-e,anubis_lib_entry). Reached only by
 * executing the .so directly; dlopen() ignores it.
 */
__attribute__((noreturn, no_stack_protector, used))
void anubis_lib_entry(void)
{
    static const char msg[] =
        "\n"
        "  libanubis_solver.so is a shared library, not an executable.\n"
        "\n"
        "  It has no main() and is not meant to be run. Load it from a program\n"
        "  instead, for example:\n"
        "\n"
        "      python3 -c \"import anubis_solver as a; print(a.AnubisSolver().backends())\"\n"
        "      python3 anubis_solver.py --backends\n"
        "      ./anubis_api            (after `make`)\n"
        "\n"
        "  Made by celikfana and DeepSeek V4.1 Flash.\n"
        "  See README.md for the full story.\n"
        "\n";

    anubis_write_all(2, msg, sizeof(msg) - 1);
    anubis_raw_exit(1);
}
