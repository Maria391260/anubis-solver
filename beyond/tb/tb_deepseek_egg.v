`timescale 1ns/1ps
// tb_deepseek_egg.v — the secret call in BAR0 returns what it promises.
//
// Reads the whole window as a host would, rebuilds the UTF-8 from the 32-bit
// little-endian words and compares it with the expected text. Three instances
// cover the three ALT_MODE settings:
//
//   main  ALT_MODE=2  always the DeepSeek credit (the default text)
//   alt   ALT_MODE=1  always the joke
//   coin  ALT_MODE=0  the 1-in-1000 coin; the sweep must be exactly one of the
//                     two texts, not a mixture, and must not change between
//                     two sweeps (the choice is latched until reset).
//
// ALT_MODE=0 is the chip. The coin is primed with one read and a clock edge
// before the sweep because a host captures the data at the AXI clock edge,
// while this bench samples a combinational output; the priming read emulates
// that edge so the first word and the rest use the same latched choice.
//
// It also proves the window does not swallow neighbouring registers: hit must
// be low for the ID, the digest area, the signature's neighbours and
// everything past the signature.
module tb_deepseek_egg;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg  [31:0] addr_m = 0, addr_a = 0, addr_r = 0;
    wire [31:0] rdata_m, rdata_a, rdata_r;
    wire        hit_m, hit_a, hit_r;

    deepseek_egg #(.ALT_MODE(2'd2)) dut_main (
        .clk(clk), .rst_n(rst_n), .addr(addr_m), .rdata(rdata_m), .hit(hit_m));
    deepseek_egg #(.ALT_MODE(2'd1)) dut_alt (
        .clk(clk), .rst_n(rst_n), .addr(addr_a), .rdata(rdata_a), .hit(hit_a));
    deepseek_egg #(.ALT_MODE(2'd0)) dut_coin (
        .clk(clk), .rst_n(rst_n), .addr(addr_r), .rdata(rdata_r), .hit(hit_r));

    // The expected texts come from the generator, via vectors/.
    localparam VEC_MAIN = "vectors/deepseek_egg.txt";
    localparam VEC_ALT  = "vectors/deepseek_egg_alt.txt";

    integer errors = 0;
    integer i, j;

    task probe(input integer which, input [31:0] a);
        begin
            case (which)
                0: addr_m = a;
                1: addr_a = a;
                2: addr_r = a;
            endcase
            #1;
        end
    endtask

    function [31:0] rd(input integer which);
        begin
            case (which)
                0: rd = rdata_m;
                1: rd = rdata_a;
                default: rd = rdata_r;
            endcase
        end
    endfunction

    function hit_at(input integer which);
        begin
            case (which)
                0: hit_at = hit_m;
                1: hit_at = hit_a;
                default: hit_at = hit_r;
            endcase
        end
    endfunction

    // Is the address inside the message window (0x060..0x0F3)?
    function in_window(input [31:0] a);
        in_window = (a >= 32'h060) && (a <= 32'h0F3);
    endfunction

    // Byte i of a little-endian 32-bit word.
    function [7:0] byte_at(input [31:0] w, input integer b);
        byte_at = w[8*b +: 8];
    endfunction

    reg [7:0] text [0:511];
    reg [7:0] want_main [0:511];
    reg [7:0] want_alt  [0:511];
    integer wn_main, wn_alt, tn;
    integer fd, ch;

    // Sweep the 37-word window, stop at the first NUL and compare with the
    // text at index `vec` (0 = credit, 1 = joke).
    task check_msg(input integer which, input integer vec);
        integer k, m, n, ok;
        begin
            m = 0;
            for (k = 0; k < 37; k = k + 1) begin
                probe(which, 32'h060 + 4*k);
                if (hit_at(which) !== 1'b1) begin
                    $display("FAIL: 0x%03x inside window but hit low", 32'h060 + 4*k);
                    errors = errors + 1;
                end
                for (j = 0; j < 4; j = j + 1) begin
                    text[m] = byte_at(rd(which), j);
                    m = m + 1;
                end
            end
            n = 0;
            while (n < m && text[n] != 8'h00) n = n + 1;
            if (vec == 0) tn = wn_main;
            else          tn = wn_alt;
            ok = 1;
            if (n != tn) begin
                $display("FAIL: mode %0d length %0d, vector %0d has %0d",
                         which, n, vec, tn);
                errors = errors + 1;
                ok = 0;
            end
            for (k = 0; k < n && k < tn; k = k + 1) begin
                if (vec == 0) ch = want_main[k];
                else          ch = want_alt[k];
                if (text[k] !== ch[7:0]) begin
                    $display("FAIL: mode %0d byte %0d = %02x, want %02x",
                             which, k, text[k], ch[7:0]);
                    errors = errors + 1;
                    ok = 0;
                end
            end
            if (ok) begin
                $write("  mode %0d: ", which);
                for (k = 0; k < n; k = k + 1) $write("%c", text[k]);
                $write("\n");
            end
        end
    endtask

    // Same sweep, but accept either text. Returns 0 for the credit and 1 for
    // the joke, or -1 on a mixture/no match.
    task check_any(input integer which, output integer got);
        integer k, m, n, ok_main, ok_alt;
        begin
            m = 0;
            for (k = 0; k < 37; k = k + 1) begin
                probe(which, 32'h060 + 4*k);
                for (j = 0; j < 4; j = j + 1) begin
                    text[m] = byte_at(rd(which), j);
                    m = m + 1;
                end
            end
            n = 0;
            while (n < m && text[n] != 8'h00) n = n + 1;
            ok_main = (n == wn_main);
            ok_alt  = (n == wn_alt);
            for (k = 0; k < n; k = k + 1) begin
                if (ok_main && text[k] !== want_main[k]) ok_main = 0;
                if (ok_alt  && text[k] !== want_alt[k])  ok_alt  = 0;
            end
            if (ok_main && !ok_alt)      got = 0;
            else if (ok_alt && !ok_main) got = 1;
            else begin
                got = -1;
                $display("FAIL: the coin returned neither whole text (len %0d, main=%0d alt=%0d)",
                         n, ok_main, ok_alt);
                errors = errors + 1;
            end
        end
    endtask

    integer got1, got2;

    initial begin
        rst_n = 0;
        #20;
        rst_n = 1;
        @(posedge clk);
        #1;

        // ---- vectors ------------------------------------------------------
        fd = $fopen(VEC_MAIN, "r");
        if (fd == 0) begin
            $display("FAIL: cannot open %s (run from the beyond/ directory)", VEC_MAIN);
            errors = errors + 1;
            wn_main = 0;
        end else begin
            wn_main = 0;
            ch = $fgetc(fd);
            while (ch != -1 && wn_main < 512) begin
                want_main[wn_main] = ch[7:0];
                wn_main = wn_main + 1;
                ch = $fgetc(fd);
            end
            $fclose(fd);
        end
        fd = $fopen(VEC_ALT, "r");
        if (fd == 0) begin
            $display("FAIL: cannot open %s (run from the beyond/ directory)", VEC_ALT);
            errors = errors + 1;
            wn_alt = 0;
        end else begin
            wn_alt = 0;
            ch = $fgetc(fd);
            while (ch != -1 && wn_alt < 512) begin
                want_alt[wn_alt] = ch[7:0];
                wn_alt = wn_alt + 1;
                ch = $fgetc(fd);
            end
            $fclose(fd);
        end
        if (wn_main != 145) begin
            $display("FAIL: credit vector is %0d bytes, want 145", wn_main);
            errors = errors + 1;
        end
        if (wn_alt != 129) begin
            $display("FAIL: joke vector is %0d bytes, want 129", wn_alt);
            errors = errors + 1;
        end

        // ---- the signature register, in every mode ------------------------
        for (i = 0; i < 3; i = i + 1) begin
            probe(i, 32'h0FC);
            if (rd(i) !== 32'hDEED5EED) begin
                $display("FAIL: mode %0d SIG = %08x", i, rd(i));
                errors = errors + 1;
            end
            if (hit_at(i) !== 1'b1) begin
                $display("FAIL: mode %0d SIG not flagged as hit", i);
                errors = errors + 1;
            end
        end

        // ---- addresses that must not be captured --------------------------
        probe(0, 32'h000); if (hit_m !== 1'b0) begin $display("FAIL: ID captured"); errors = errors + 1; end
        probe(0, 32'h040); if (hit_m !== 1'b0) begin $display("FAIL: digest captured"); errors = errors + 1; end
        probe(0, 32'h05C); if (hit_m !== 1'b0) begin $display("FAIL: digest tail captured"); errors = errors + 1; end
        probe(0, 32'h0F4); if (hit_m !== 1'b0) begin $display("FAIL: word after message captured"); errors = errors + 1; end
        probe(0, 32'h0F8); if (hit_m !== 1'b0) begin $display("FAIL: gap captured"); errors = errors + 1; end
        probe(0, 32'h100); if (hit_m !== 1'b0) begin $display("FAIL: above window captured"); errors = errors + 1; end

        // ---- both texts ---------------------------------------------------
        check_msg(0, 0);
        check_msg(1, 1);

        // ---- the coin: one whole text, stable across two sweeps -----------
        // Prime the latch the way an AXI read captures it at a clock edge.
        probe(2, 32'h060);
        @(posedge clk);
        #1;
        check_any(2, got1);
        check_any(2, got2);
        if (got1 >= 0 && got1 != got2) begin
            $display("FAIL: the coin changed between sweeps (%0d then %0d)", got1, got2);
            errors = errors + 1;
        end else if (got1 >= 0)
            $display("  the coin chose mode %0d and held it", got1);

        // ---- last word before the signature -------------------------------
        probe(0, 32'h0F0);   // word 36 of 37: "!" then zero padding
        if (rdata_m !== 32'h0000_0021) begin
            $display("FAIL: tail word = %08x", rdata_m);
            errors = errors + 1;
        end else
            $display("  tail word %08x is '!' plus padding", rdata_m);

        if (errors == 0) begin
            $display("tb_deepseek_egg: the secret call answers");
            $finish;
        end else begin
            $display("tb_deepseek_egg: %0d FAILURES", errors);
            $fatal(1);
        end
    end
endmodule
