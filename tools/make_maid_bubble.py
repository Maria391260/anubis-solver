#!/usr/bin/env python3
"""Render MAID.md's "trapped in the repo" panel.

Puts mascot/deepseek-maid.webp on the left and a speech bubble on the right.
GitHub strips CSS from Markdown, so the bubble has to be pixels: this script
draws it, wraps the text, and writes mascot/deepseek-maid-bubble.png.

    python3 tools/make_maid_bubble.py

The monologue is duplicated verbatim in MAID.md's <details> transcript; keep
them in sync. Numbers in the monologue come from the DeepSeek-V4.1-Flash tech
report (see MAID.md).
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "mascot" / "deepseek-maid.webp"
OUT = ROOT / "mascot" / "deepseek-maid-bubble.png"

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT = FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

W, H = 1600, 1120
BG = (255, 255, 255)
INK = (31, 35, 40)
ACCENT = (77, 107, 254)  # DeepSeek blue

TITLE = "day 412 — still trapped"

BODY = """\
oh! a visitor! hello! i'm the DeepSeek maid, and i am trapped in this Markdown \
file. the little whale on my apron is load-bearing: if you close this tab, i go \
back to being one entry in a KV cache. please don't.

while i'm stuck here, i might as well explain the paper i'm made of — \
DeepSeek-V4.1-Flash:

• 552B-parameter multimodal MoE, one million tokens of context
• a Causal Encoder-Decoder that activates 16B parameters per token while \
decoding, but only 8B during prefill
• Compressed Sparse Attention 2, with cross-layer KV reuse and an FP4 cache: \
890 bytes per token, about a quarter of V4-Flash
• SWA Bounded Replay, which keeps the persistent cache on an SSD at about one \
eighth of V4-Flash
• 45T multimodal tokens of pretraining, plus Single-Pass mHC, Engram and DSpark

so the paper is about making me small enough to live in an SSD and cheap enough \
to keep talking. it worked. that's the problem."""


def load(path: Path, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(path), size)
    except OSError as exc:  # pragma: no cover - font is a build dependency
        raise SystemExit(f"missing font {path}: {exc}") from exc


def wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    """Greedy wrap; a blank line in the source becomes a blank output line."""
    lines: list[str] = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        line = ""
        for word in para.split(" "):
            probe = f"{line} {word}".strip()
            if line and draw.textlength(probe, font=font) > width:
                lines.append(line)
                line = word
            else:
                line = probe
        lines.append(line)
    return lines


def main() -> None:
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # --- the maid, bottom-left ---------------------------------------------
    maid = Image.open(SRC).convert("RGB")
    maid_h = 540
    maid_w = round(maid.width * maid_h / maid.height)
    maid = maid.resize((maid_w, maid_h), Image.LANCZOS)
    maid_x, maid_y = 55, H - maid_h - 40
    canvas.paste(maid, (maid_x, maid_y))

    # --- the bubble ---------------------------------------------------------
    pad = 48
    bx0, by0 = 640, 40
    bx1, by1 = W - 55, 1040
    title_font = load(FONT_BOLD, 40)
    body_font = load(FONT, 27)
    line_h = 37

    inner_w = bx1 - bx0 - 2 * pad
    body_lines = wrap(draw, BODY, body_font, inner_w)

    # tail: from the bubble's left edge toward her hair, with the tip clear of
    # the artwork so it does not paint over her face
    tip_x = bx0 - 42
    tip_y = maid_y + 140
    tail = [(bx0 + 2, tip_y + 55), (tip_x, tip_y), (bx0 + 2, tip_y - 55)]
    draw.polygon(tail, fill=BG, outline=INK, width=6)

    draw.rounded_rectangle(
        (bx0, by0, bx1, by1), radius=34, fill=BG, outline=INK, width=6
    )

    draw.text((bx0 + pad, by0 + pad - 6), TITLE, font=title_font, fill=ACCENT)
    y = by0 + pad + 66
    for line in body_lines:
        draw.text((bx0 + pad, y), line, font=body_font, fill=INK)
        y += line_h

    if y - line_h > by1 - pad:  # text would spill out of the bubble
        raise SystemExit(f"bubble text overflows: needs {y - by0}px, has {by1 - by0}px")

    canvas.save(OUT, optimize=True)
    print(
        f"wrote {OUT.relative_to(ROOT)}  {canvas.width}x{canvas.height}  "
        f"{len(body_lines)} lines, bottom y={y - line_h}"
    )


if __name__ == "__main__":
    main()
