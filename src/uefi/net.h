/* A blocking TCP server on top of EFI_TCP4. One connection at a time, no
 * threads: a PoW client pipelines frames anyway, and the firmware's event loop
 * is not worth fighting for a second core that does not exist.
 *
 * Framing: 4-byte big-endian length, then that many bytes of JSON. It is not
 * HTTP on purpose - HTTP would mean a parser in a 2 MB PE image for no gain. */
#pragma once

#include <efi.h>

typedef int (*net_handler)(const char *req, int len, char *resp, int cap,
                           void *ctx);

EFI_STATUS net_serve(UINT16 port, net_handler handler, void *ctx);
