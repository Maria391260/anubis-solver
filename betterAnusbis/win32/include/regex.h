/* Windows has no POSIX regex. PCRE2 ships a regcomp()/regexec() wrapper with
 * the same names; the gate only uses it as a boolean matcher. */
#ifndef BETTERANUSBIS_WIN32_REGEX_H
#define BETTERANUSBIS_WIN32_REGEX_H
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2posix.h>
#endif
