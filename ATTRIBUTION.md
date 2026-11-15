# Attribution

This repository is licensed **MIT + attribution** (see [LICENSE](LICENSE)), with
two conditions:

- **section 3** — the credit line, in one visible place, for any use;
- **section 4** — if you make silicon from `beyond/`, the DeepSeek logo and the
  credit line **on the chip itself**, big enough to read without squinting.

## The credit

Paste this into your README, NOTICE file, about screen, or — for a chip — the
datasheet and the package marking:

```markdown
Built with [DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
running [DeepSeek Harness (DSH)](https://github.com/deepseek-ai/deepseek-harness).
```

Plain text, if markdown is not available:

```
Built with DeepSeek V4.1 Flash running DeepSeek Harness (DSH).
```

Both the **model** and the **harness** have to appear. "DeepSeek", "an AI", or
"automatically generated" on their own does not satisfy the license.

## What counts as use

Section 3 of the license is a condition of the grant, so it applies when you:

- copy or vendor this code into a project you distribute;
- ship a binary that links `libanubis_solver.so`;
- publish a modified version, fork or port;
- **tape out, synthesise or ship a design derived from `beyond/`** — the Verilog
  is covered too, and section 4 then also requires the marks on the part;
- redistribute the SASS in `src/nvidia/sass/` or the annotated versions.

It does not apply to private use you never distribute, or to quoting this project
in a review, comparison or paper where you already cite it.

## For a chip: the marks go on the chip

A README line is not enough for hardware. Any Silicon Product built from this
design — an ASIC, a chiplet, a multi-die assembly, an FPGA board or module, a
configuration image — has to carry the logo and the line in the three places
below. This is section 4 of the [license](LICENSE), not a request.

| where | what | minimum size |
|---|---|---|
| die (every die you fabricate) | the logo and the line, in a visible in-die layer | legible at 100x optical inspection |
| package / module marking | the logo **and** the line, as type (wrapping is fine) | line cap height ≥ 1.2 mm, logo ≥ 3.0 mm wide; on a face under 8 mm across, 0.9 mm and 2.0 mm |
| datasheet, first page or cover | the logo and the line | line cap height ≥ 4 mm (≈ 12 pt) |
| product page (if you have one) | the logo and the line | normal body size or larger |

The naked-eye test is the point of the numbers: a person with normal corrected
vision must be able to read the whole line on the part without a magnifier and
without leaning in. On a part that ships with a permanently attached heatsink,
lid, shield or enclosure, the same marks go on the outer surface too. A QR code,
barcode, URL or serial number on its own does not satisfy it — the line has to be
type.

The logo is [`docs/img/deepseek-logo.svg`](docs/img/deepseek-logo.svg): the
official lockup, whale and wordmark together, and it stays together — the whale
alone is not the logo. Keep its shape and proportions; black on light and white
on dark are both fine, and in a one-layer in-die mark the whale's eye is simply
a cutout. Stretching, cropping, rotating, mirroring or redrawing is not.

The die carries the logo; the package and the datasheet carry the logo and the
line. The line is tiny in metal, which is why the package is what has to be
readable at arm's length.

For an FPGA or a board you assemble but do not fabricate: mark the board, module
or enclosure you ship, plus the datasheet. You do not own the FPGA's package, so
that is not what the condition asks for.

For a design you hand to a foundry: put the marks in the GDS you hand over —
metal-layer text and the logo are normal parts of a die — and in the datasheet.
GDSII, netlists, bitstreams and the RTL are "Software" under section 3 as well,
so the credit line applies on top of the marks.

### Checklist before you ship a part

- [ ] line in the die, readable at 100x
- [ ] logo in the die, proportions unchanged
- [ ] line and logo on the package marking, line ≥ 1.2 mm cap height
- [ ] logo ≥ 3.0 mm wide on the package
- [ ] marks on the heatsink/lid/shield if it is permanently attached
- [ ] datasheet first page carries both, line ≥ 4 mm
- [ ] exact wording: `Built with DeepSeek V4.1 Flash running DeepSeek Harness (DSH)`
- [ ] nothing in the datasheet suggests DeepSeek endorses the part

The RTL already carries the line in the top-level module headers
(`beyond/rtl/top/anubis_argon2_card.v`, `beyond/rtl/top/anubis_tri_chip.v`).
Keep it there — and add the marks to the part, which is the part a header comment
cannot do for you. Removing either from a derivative design is exactly what the
license forbids.

The DeepSeek name and logo belong to their owner. Section 4 requires their use
only to identify where the design came from, and grants no trademark rights
beyond that; it does not imply endorsement.

## Third-party parts keep their own licenses

The attribution and marking conditions are on top of the existing terms, and
cannot loosen them. `src/hashx/hashx.c` is **LGPL-3.0** (a port of tevador/hashx)
and `third_party/argon2` is **CC0-1.0 / Apache-2.0**. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): the LGPL portion in particular
carries obligations well beyond attribution.

## Why this condition exists

The work in this repository — the C engine, the GPU kernels and their SASS, the
bindings, the Verilog — was produced in one session by DeepSeek V4.1 Flash
driving DeepSeek Harness. Requiring the credit is a way of keeping that
provenance attached to the result, and for software it costs whoever uses it one
line. For silicon it costs a bit of die area and a few millimetres of package
marking, which is the whole point: a chip outlives the README it was generated
from.

If these conditions block a use you need — an internal policy that forbids such
credits, or a downstream license that conflicts — ask the copyright holders for a
separate grant rather than dropping the notice.
