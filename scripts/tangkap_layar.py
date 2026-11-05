"""
scripts/tangkap_layar.py - Mengambil screenshot setiap tab untuk README (butuh Playwright + Chromium).

    streamlit run app.py --server.port 8513 --server.headless true &
    python scripts/tangkap_layar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8513"
KELUARAN = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
TAB = [
    ("Radar", "01_radar.png"),
    ("Topic clusters", "02_topik.png"),
    ("Keyword trends", "03_tren_kata.png"),
    ("Similarity check", "04_kemiripan.png"),
    ("Modus", "05_modus.png"),
    ("Data & Sources", "06_data.png"),
    ("How it works", "07_cara_kerja.png"),
]


def main() -> None:
    KELUARAN.mkdir(parents=True, exist_ok=True)
    exe = sys.argv[1] if len(sys.argv) > 1 else None
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto(URL, wait_until="networkidle", timeout=120_000)
        page.wait_for_selector("[role='tab']", timeout=180_000)
        page.wait_for_timeout(4000)
        for nama, berkas in TAB:
            page.get_by_role("tab", name=nama).click()
            page.wait_for_timeout(3500)
            page.screenshot(path=str(KELUARAN / berkas))
            print("ok", berkas)
        page.get_by_role("tab", name="Topic clusters").click()
        page.get_by_text("Learning mode").click()
        page.wait_for_timeout(2500)
        page.locator("summary:visible").first.click()
        page.wait_for_timeout(1500)
        page.screenshot(path=str(KELUARAN / "08_learning_mode.png"))
        print("ok 08_learning_mode.png")
        browser.close()


if __name__ == "__main__":
    main()
