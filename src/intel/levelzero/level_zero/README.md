# Vendored Level Zero headers

`ze_api.h` is Intel's oneAPI Level Zero API header, copied verbatim so the
universal build can compile the Level Zero backend anywhere — no `libze-dev`
installed, no root, cross-compiles included. It is the same reasoning as
`src/google/tpu/pjrt_c_api.h`: the runtime loader is `dlopen()`ed, and the
header carries no code.

| | |
|---|---|
| file | `ze_api.h` |
| version | Level Zero 1.24.1 (package `libze-dev` 1.24.1-2, Ubuntu questing) |
| upstream | <https://github.com/oneapi-src/level-zero> |
| license | MIT, `Copyright (C) 2019-2021 Intel Corporation` |
| license note | every file of the project is MIT; see `THIRD_PARTY_NOTICES.md` |

Do not edit it. Refresh it by copying
`/usr/include/level_zero/ze_api.h` from a `libze-dev` package of the wanted
version, and update the version row above.

`Makefile` uses this copy when it is present and falls back to
`/usr/include/level_zero/ze_api.h`, then to `/usr/local/include`, so a system
header of a newer version still wins if you point `L0_INC` at it explicitly.
