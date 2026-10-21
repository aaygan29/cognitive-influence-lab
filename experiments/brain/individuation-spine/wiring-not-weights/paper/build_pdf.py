"""
Render manuscript.md -> manuscript.pdf, pure-Python (no LaTeX / pandoc / system libs).
Uses fpdf2. Handles headings, paragraphs, blockquotes, lists, and embedded figures.
Usage: python3 build_pdf.py   (pip install fpdf2)
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "manuscript.md")
OUT = os.path.join(HERE, "manuscript.pdf")

# map non-latin-1 glyphs to safe ASCII so the core font always renders
SUBS = {"–": "-", "—": "-", "≈": "~", "→": "->", "×": "x", "·": ".", "’": "'", "‘": "'",
        "“": '"', "”": '"', "α": "alpha", "β": "beta", "Φ": "Phi", "σ": "sigma", "Δ": "d",
        "≥": ">=", "≤": "<=", "…": "...", "•": "-"}


def clean(s):
    for k, v in SUBS.items():
        s = s.replace(k, v)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)   # drop bold markers
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*", r"\1", s)  # drop italic markers
    s = s.replace("`", "")
    return s.encode("latin-1", "replace").decode("latin-1")


def main():
    try:
        from fpdf import FPDF
    except ImportError:
        raise SystemExit("fpdf2 not installed. Run: pip install fpdf2")

    pdf = FPDF(format="letter")
    pdf.set_auto_page_break(True, margin=18)
    pdf.set_margins(20, 18, 20)
    pdf.add_page()
    W = pdf.w - 40

    lines = open(SRC).read().split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        i += 1
        if ln.strip() in ("---", ""):
            if ln.strip() == "---":
                pdf.ln(2); pdf.set_draw_color(150); pdf.line(20, pdf.get_y(), pdf.w - 20, pdf.get_y()); pdf.ln(3)
            else:
                pdf.ln(2)
            continue
        # images: ![caption](path)
        m = re.match(r"!\[(.*?)\]\((.+?)\)", ln)
        if m:
            cap, path = m.group(1), m.group(2)
            if not os.path.isabs(path):
                path = os.path.join(HERE, path)
            if os.path.exists(path):
                iw = min(150, W)
                x = (pdf.w - iw) / 2
                try:
                    pdf.image(path, x=x, w=iw)
                except Exception:
                    pass
                pdf.set_font("Helvetica", "I", 8); pdf.set_text_color(90)
                pdf.multi_cell(0, 4, clean(cap)); pdf.set_text_color(0); pdf.ln(2)
            continue
        # headings
        if ln.startswith("# "):
            pdf.set_font("Helvetica", "B", 15); pdf.multi_cell(0, 7, clean(ln[2:])); pdf.ln(1); continue
        if ln.startswith("## "):
            pdf.ln(2); pdf.set_font("Helvetica", "B", 12); pdf.multi_cell(0, 6, clean(ln[3:])); pdf.ln(1); continue
        if ln.startswith("### "):
            pdf.set_font("Helvetica", "B", 11); pdf.multi_cell(0, 5, clean(ln[4:])); continue
        # blockquote
        if ln.startswith(">"):
            pdf.set_font("Helvetica", "I", 10); pdf.set_text_color(40)
            pdf.set_x(28); pdf.multi_cell(W - 8, 5, clean(ln.lstrip("> ").strip()))
            pdf.set_text_color(0); continue
        # bullet / table row -> plain
        if ln.startswith("- ") or ln.startswith("|"):
            pdf.set_font("Helvetica", "", 9.5)
            pdf.multi_cell(0, 5, ("  - " + clean(ln[2:])) if ln.startswith("- ") else clean(ln.strip("|")))
            continue
        # normal paragraph (gather wrapped markdown line)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, clean(ln))

    pdf.output(OUT)
    print("wrote", OUT, f"({os.path.getsize(OUT)//1024} KB, {pdf.page_no()} pages)")


if __name__ == "__main__":
    main()
