/* Just enough JSON to read a challenge and write an answer. No allocations, no
 * recursion, no escapes beyond \" and \\ - a firmware has no room for a parser
 * library and no need for one. */
#pragma once

#include <stddef.h>

/* Returns 0 and fills out[] on success, -1 if the key is missing. Writes at most
 * n-1 bytes plus a terminator, and stops at a '"' or the end of the object. */
int json_str(const char *js, const char *key, char *out, size_t n);

/* Same, for a number. Returns def if the key is missing. */
long long json_int(const char *js, const char *key, long long def);

/* Length of the object/array starting at p (which must point at '{' or '['),
 * including the brackets; 0 if unbalanced. Used to detect truncated frames. */
size_t json_extent(const char *p, size_t n);
