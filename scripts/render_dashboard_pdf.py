"""
scripts/render_dashboard_pdf.py

Renders dashboard.html (the self-contained results dashboard at the repo root)
to a single-page PDF at dashboard.pdf, so the results can be viewed directly
in any PDF viewer or inline on GitHub without opening HTML.

Requires playwright + chromium:
    pip install playwright && python -m playwright install chromium

Usage:
    python scripts/render_dashboard_pdf.py
"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = PROJECT_ROOT / "dashboard.html"
PDF_PATH = PROJECT_ROOT / "dashboard.pdf"
VIEWPORT_WIDTH = 1400


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": VIEWPORT_WIDTH, "height": 900})
        page.goto(HTML_PATH.as_uri(), wait_until="load", timeout=60_000)
        page.wait_for_timeout(1500)
        height = page.evaluate("document.documentElement.scrollHeight")
        page.pdf(
            path=str(PDF_PATH),
            width=f"{VIEWPORT_WIDTH}px",
            height=f"{height + 40}px",
            print_background=True,
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
        )
        browser.close()
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
