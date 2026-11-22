#include <efi.h>
#include <efilib.h>
#include <string.h>
#include <efitcp.h>

#include "net.h"
#include "shim.h"

/* gnu-efi does not export the TCP4 GUIDs (EDK2 does), so define them from the
 * brace initialisers in efitcp.h. */
static EFI_GUID Tcp4SbGuid = EFI_TCP4_SERVICE_BINDING_PROTOCOL;
static EFI_GUID Tcp4Guid = EFI_TCP4_PROTOCOL;

#define RX_CAP 8192
#define TX_CAP 8192

static EFI_TCP4 *listener;

static EFI_TCP4 *tcp4_child(EFI_HANDLE h) {
    EFI_TCP4 *p = NULL;

    if (BS->HandleProtocol(h, &Tcp4Guid, (void **)&p) != EFI_SUCCESS) {
        return NULL;
    }
    return p;
}

static int wait_token(EFI_TCP4_COMPLETION_TOKEN *t) {
    UINTN idx;

    if (t->Event == NULL) {
        return -1;
    }
    BS->WaitForEvent(1, &t->Event, &idx);
    BS->CloseEvent(t->Event);
    t->Event = NULL;
    return EFI_ERROR(t->Status) ? (int)(t->Status & 0x7fffffff) : 0;
}

static int tcp_send(EFI_TCP4 *s, const void *buf, UINT32 len) {
    EFI_TCP4_IO_TOKEN tok;
    EFI_TCP4_TRANSMIT_DATA tx;
    EFI_TCP4_FRAGMENT_DATA frag;
    EFI_STATUS st;

    memset(&tok, 0, sizeof tok);
    memset(&tx, 0, sizeof tx);
    frag.FragmentLength = len;
    frag.FragmentBuffer = (void *)buf;
    tx.Push = TRUE;
    tx.DataLength = len;
    tx.FragmentCount = 1;
    tx.FragmentTable[0] = frag;
    tok.Packet.TxData = &tx;
    st = BS->CreateEvent(0, 0, NULL, NULL, &tok.CompletionToken.Event);
    if (EFI_ERROR(st)) {
        return -1;
    }
    st = s->Transmit(s, &tok);
    if (EFI_ERROR(st)) {
        BS->CloseEvent(tok.CompletionToken.Event);
        return -1;
    }
    return wait_token(&tok.CompletionToken);
}

static int tcp_recv(EFI_TCP4 *s, void *buf, UINT32 cap, UINT32 *got) {
    EFI_TCP4_IO_TOKEN tok;
    EFI_TCP4_RECEIVE_DATA rx;
    EFI_TCP4_FRAGMENT_DATA frag;
    EFI_STATUS st;

    memset(&tok, 0, sizeof tok);
    memset(&rx, 0, sizeof rx);
    frag.FragmentLength = cap;
    frag.FragmentBuffer = buf;
    rx.DataLength = cap;
    rx.FragmentCount = 1;
    rx.FragmentTable[0] = frag;
    tok.Packet.RxData = &rx;
    st = BS->CreateEvent(0, 0, NULL, NULL, &tok.CompletionToken.Event);
    if (EFI_ERROR(st)) {
        return -1;
    }
    st = s->Receive(s, &tok);
    if (EFI_ERROR(st)) {
        BS->CloseEvent(tok.CompletionToken.Event);
        return -1;
    }
    if (wait_token(&tok.CompletionToken) != 0) {
        return -1;
    }
    *got = rx.DataLength;
    return 0;
}

static int read_exact(EFI_TCP4 *s, void *buf, UINT32 len) {
    UINT8 *p = buf;
    UINT32 got = 0, n = 0;

    while (n < len) {
        if (tcp_recv(s, p + n, len - n, &got) != 0 || got == 0) {
            return -1;
        }
        n += got;
    }
    return 0;
}

static int handle_conn(EFI_TCP4 *s, net_handler handler, void *ctx) {
    static UINT8 rx[4 + RX_CAP];
    static char tx[4 + TX_CAP];
    UINT32 len, be;
    int r;

    for (;;) {
        if (read_exact(s, rx, 4) != 0) {
            return 0;
        }
        be = ((UINT32)rx[0] << 24) | ((UINT32)rx[1] << 16) |
             ((UINT32)rx[2] << 8) | (UINT32)rx[3];
        if (be == 0 || be > RX_CAP) {
            return 0;
        }
        len = be;
        if (read_exact(s, rx + 4, len) != 0) {
            return 0;
        }
        r = handler((const char *)rx + 4, (int)len, tx + 4, TX_CAP, ctx);
        if (r < 0) {
            return 0;
        }
        tx[0] = (char)((r >> 24) & 0xff);
        tx[1] = (char)((r >> 16) & 0xff);
        tx[2] = (char)((r >> 8) & 0xff);
        tx[3] = (char)(r & 0xff);
        if (tcp_send(s, tx, (UINT32)(r + 4)) != 0) {
            return 0;
        }
    }
}

EFI_STATUS net_serve(UINT16 port, net_handler handler, void *ctx) {
    EFI_HANDLE *handles = NULL;
    UINTN count = 0, i;
    EFI_STATUS st;
    EFI_TCP4_CONFIG_DATA cfg;
    EFI_TCP4_OPTION opt;
    EFI_SERVICE_BINDING *sb = NULL;

    st = BS->LocateHandleBuffer(ByProtocol, &Tcp4SbGuid,
                                NULL, &count, &handles);
    if (EFI_ERROR(st) || count == 0) {
        anubis_console_write("net: no EFI_TCP4 service binding\n");
        return EFI_UNSUPPORTED;
    }
    for (i = 0; i < count && listener == NULL; i++) {
        EFI_HANDLE child = NULL;
        EFI_TCP4 *p = NULL;

        st = BS->HandleProtocol(handles[i], &Tcp4SbGuid, (void **)&sb);
        if (EFI_ERROR(st)) {
            continue;
        }
        st = sb->CreateChild(sb, &child);
        if (EFI_ERROR(st) || child == NULL) {
            continue;
        }
        st = BS->HandleProtocol(child, &Tcp4Guid, (void **)&p);
        if (EFI_ERROR(st) || p == NULL) {
            continue;
        }
        listener = p;
    }
    if (listener == NULL) {
        anubis_console_write("net: no usable TCP4 instance\n");
        return EFI_UNSUPPORTED;
    }

    memset(&opt, 0, sizeof opt);
    opt.ReceiveBufferSize = 0x40000;
    opt.SendBufferSize = 0x40000;
    opt.MaxSynBackLog = 4;
    opt.ConnectionTimeout = 10;   /* seconds; 0 means "never give up" */
    opt.DataRetries = 5;
    opt.FinTimeout = 4;
    opt.TimeWaitTimeout = 8;
    opt.KeepAliveProbes = 0;
    opt.EnableNagle = TRUE;

    memset(&cfg, 0, sizeof cfg);
    cfg.TypeOfService = 0;
    cfg.TimeToLive = 64;
    cfg.AccessPoint.UseDefaultAddress = TRUE;
    cfg.AccessPoint.StationPort = port;
    cfg.AccessPoint.ActiveFlag = FALSE;
    cfg.ControlOption = &opt;

    st = listener->Configure(listener, &cfg);
    if (EFI_ERROR(st)) {
        anubis_console_write("net: Configure() failed\n");
        return st;
    }
    anubis_console_write("net: listening\n");

    for (;;) {
        EFI_TCP4_LISTEN_TOKEN lt;
        EFI_TCP4 *conn = NULL;
        EFI_TCP4_CLOSE_TOKEN ct;

        memset(&lt, 0, sizeof lt);
        st = BS->CreateEvent(0, 0, NULL, NULL, &lt.CompletionToken.Event);
        if (EFI_ERROR(st)) {
            return st;
        }
        st = listener->Accept(listener, &lt);
        if (EFI_ERROR(st)) {
            BS->CloseEvent(lt.CompletionToken.Event);
            continue;
        }
        if (wait_token(&lt.CompletionToken) != 0) {
            anubis_console_write("net: accept failed\n");
            continue;
        }
        if (lt.NewChildHandle != NULL) {
            conn = tcp4_child(lt.NewChildHandle);
        }
        if (conn == NULL) {
            conn = listener;
        }
        anubis_console_write("net: connection\n");
        handle_conn(conn, handler, ctx);
        anubis_console_write("net: connection closed\n");

        if (conn != listener) {
            memset(&ct, 0, sizeof ct);
            BS->CreateEvent(0, 0, NULL, NULL, &ct.CompletionToken.Event);
            conn->Close(conn, &ct);
            wait_token(&ct.CompletionToken);
        }
    }
}
