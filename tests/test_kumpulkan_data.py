"""Tes pengurai halaman Turnbackhoax di scripts/kumpulkan_data.py, memakai potongan HTML tiruan (tanpa internet)."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts" / "kumpulkan_data.py"
spec = importlib.util.spec_from_file_location("kumpulkan_data", PATH)
kd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kd)

KARTU = """
<div class="news-card-h-alt flex">
  <a href="https://turnbackhoax.id/articles/12345-salah-video-banjir"><img src="x.jpg" alt="[SALAH] Video Banjir &amp; Longsor di Jakarta"></a>
  <div><a href="/articles?category=politik&page=1">Politik</a>
    <span class="hidden xl:block">Video <b>lama</b>   dari 2019.</span>
    <span>05/11/2026</span>
  </div>
</div>
</div>
</div>
"""
KARTU_TANPA_URL = '<div class="news-card-h-alt"><img alt="x"></div>\n</div>\n</div>'


def test_bersihkan_buang_tag_dan_spasi():
    assert kd.bersihkan("  Video <b>lama</b>\n  dari &amp; ke ") == "Video lama dari & ke"


@pytest.mark.parametrize("judul, label", [
    ("[SALAH] Video banjir", "SALAH"),
    ("(Hoaks) Pesan berantai", "HOAKS"),
    ("[ Misleading ] Foto lama", "MISLEADING"),
    ("Klarifikasi soal bansos", "TANPA LABEL"),
])
def test_tebak_label(judul, label):
    assert kd.tebak_label(judul) == label


def test_urai_halaman_satu_kartu():
    (b,) = kd.urai_halaman(KARTU + KARTU_TANPA_URL)
    assert b["id"] == 12345
    assert b["tanggal"] == "2026-11-05"
    assert b["judul"] == "[SALAH] Video Banjir & Longsor di Jakarta"
    assert b["label"] == "SALAH"
    assert b["kategori"] == "Politik"
    assert b["ringkasan"] == "Video lama dari 2019."
    assert b["url"].endswith("12345-salah-video-banjir")


def test_buang_tanggal_mustahil():
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5],
                       "tanggal": ["2020-05-01", "1000-01-01", "6325-02-02", "", "2013-12-31"]})
    assert kd.buang_tanggal_mustahil(df)["id"].tolist() == [1]


def test_simpan_buang_duplikat_dan_urutkan(monkeypatch, tmp_path):
    keluar = tmp_path / "data" / "hoaks.csv"
    monkeypatch.setattr(kd, "BERKAS_KELUARAN", keluar)
    baris = [{k: "" for k in kd.KOLOM} | {"id": i, "tanggal": t} for i, t in
             [(1, "2024-01-01"), (2, "2025-06-01"), (2, "2025-06-01"), (3, "1000-01-01")]]
    kd.simpan(pd.DataFrame(baris), senyap=True)
    hasil = pd.read_csv(keluar)
    assert hasil["id"].tolist() == [2, 1]
    assert list(hasil.columns) == kd.KOLOM
