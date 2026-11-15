`timescale 1ns/1ps
// tb_deepseek_egg.v — the secret call in BAR0 returns what it promises.
//
// Reads the whole window as a host would, rebuilds the ASCII from the 32-bit
// little-endian words and compares it with the expected sentence. It also
// proves the window does not swallow neighbouring registers: hit must be low
// for the ID, the digest area, the signature's neighbours and everything past
// 0x0FC.
module tb_deepseek_egg;
    reg  [31:0] addr = 0;
    wire [31:0] rdata;
    wire        hit;

    deepseek_egg dut (.addr(addr), .rdata(rdata), .hit(hit));

    // The expected sentence comes from the generator, via vectors/.
    localparam VEC = "vectors/deepseek_egg.txt";

    integer errors = 0;
    integer i;

    task probe(input [31:0] a);
        begin
            addr = a;
            #1;
        end
    endtask

    // Is the address inside the message window (0x060..0x0F3)?
    function in_window(input [31:0] a);
        in_window = (a >= 32'h060) && (a <= 32'h0F3);
    endfunction

    // Byte i of a little-endian 32-bit word.
    function [7:0] byte_at(input [31:0] w, input integer b);
        byte_at = w[8*b +: 8];
    endfunction

    reg [7:0] text [0:511];
    reg [7:0] want [0:511];
    integer n, wn;
    integer fd, ch;
    integer j;

    initial begin
        // ---- the signature register ---------------------------------------
        probe(32'h0FC);
        if (rdata !== 32'hDEED5EED) begin
            $display("FAIL: SIG = %08x, want DEED5EED", rdata);
            errors = errors + 1;
        end else
            $display("  SIG reads %08x", rdata);
        if (hit !== 1'b1) begin $display("FAIL: SIG not flagged as hit"); errors = errors + 1; end

        // ---- addresses that must not be captured --------------------------
        probe(32'h000); if (hit !== 1'b0) begin $display("FAIL: ID captured"); errors = errors + 1; end
        probe(32'h040); if (hit !== 1'b0) begin $display("FAIL: digest captured"); errors = errors + 1; end
        probe(32'h05C); if (hit !== 1'b0) begin $display("FAIL: digest tail captured"); errors = errors + 1; end
        probe(32'h0F4); if (hit !== 1'b0) begin $display("FAIL: word after message captured"); errors = errors + 1; end
        probe(32'h0F8); if (hit !== 1'b0) begin $display("FAIL: gap captured"); errors = errors + 1; end
        probe(32'h100); if (hit !== 1'b0) begin $display("FAIL: above window captured"); errors = errors + 1; end

        // ---- walk the message ---------------------------------------------
        n = 0;
        for (i = 0; i < 64; i = i + 1) begin
            probe(32'h060 + 4 * i);
            if (in_window(32'h060 + 4 * i) && hit !== 1'b1) begin
                $display("FAIL: 0x%03x inside window but hit low", 32'h060 + 4 * i);
                errors = errors + 1;
            end
            for (j = 0; j < 4; j = j + 1)
                text[n + j] = byte_at(rdata, j);
            n = n + 4;
        end

        // The message is zero padded, so the first NUL terminates it.
        i = 0;
        while (i < n && text[i] != 8'h00) i = i + 1;
        if (i == 0) begin $display("FAIL: message is empty"); errors = errors + 1; end
        if (i != 145) begin
            $display("FAIL: message length %0d, want 145", i);
            errors = errors + 1;
        end

        // Compare byte for byte against the sentence in vectors/.
        fd = $fopen(VEC, "r");
        if (fd == 0) begin
            $display("FAIL: cannot open %s (run from the beyond/ directory)", VEC);
            errors = errors + 1;
            wn = 0;
        end else begin
            wn = 0;
            ch = $fgetc(fd);
            while (ch != -1 && wn < 512) begin
                want[wn] = ch[7:0];
                wn = wn + 1;
                ch = $fgetc(fd);
            end
            $fclose(fd);
        end
        if (wn != i) begin
            $display("FAIL: message length %0d, vector has %0d", i, wn);
            errors = errors + 1;
        end
        for (j = 0; j < i && j < wn; j = j + 1) begin
            if (text[j] !== want[j]) begin
                $display("FAIL: byte %0d = '%c' (%02x), want '%c' (%02x)",
                         j, text[j], text[j], want[j], want[j]);
                errors = errors + 1;
            end
        end
        $write("  message: ");
        for (j = 0; j < i; j = j + 1) $write("%c", text[j]);
        $write("\n");

        // ---- last word before the signature, and the wrap behaviour -------
        probe(32'h0F0);   // word 36 of 37: "!" then zero padding
        if (rdata !== 32'h0000_0021) begin
            $display("FAIL: tail word = %08x", rdata);
            errors = errors + 1;
        end else
            $display("  tail word %08x is '!' plus padding", rdata);

        if (errors == 0) begin
            $display("tb_deepseek_egg: the secret call answers");
            $finish;
        end else begin
            $display("tb_deepseek_egg: %0d FAILURES", errors);
            $fatal(1);
        end
    end
endmodule
