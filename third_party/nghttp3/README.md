# nghttp3 (vendored static build)

nghttp3 is the HTTP/3 framing and QPACK implementation that betterAnusbis
uses together with OpenSSL's QUIC stack. Neither OpenSSL nor an HTTP/3
implementation is otherwise part of a base system, and a gate that cannot be
built without a network is a gate nobody builds behind a firewall. This
directory carries a Release, library-only, static build so `make` works
offline.

- Upstream: https://github.com/ngtcp2/nghttp3
- Version: 1.8.0
- License: MIT (see `COPYING` in this directory)
- Built with:
  ```sh
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_SHARED_LIB=OFF -DENABLE_STATIC_LIB=ON \
        -DENABLE_LIB_ONLY=ON -DCMAKE_INSTALL_PREFIX=/tmp/nghttp3
  cmake --build build && cmake --install build
  cp -r /tmp/nghttp3/include/nghttp3 third_party/nghttp3/include/
  cp /tmp/nghttp3/lib/libnghttp3.a third_party/nghttp3/lib/
  ```
  The static library is x86-64 here. On another architecture, rebuild it with
  the commands above; `make NGHTTP3_PREFIX=/path/to/nghttp3` picks it up, and
  a `libnghttp3.pc` on the system is found through pkg-config as well.
  `make` without nghttp3 still builds and serves HTTP/1.1 and HTTP/2.
