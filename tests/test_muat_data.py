"""Uji pemuatan data dan RSS (src/muat_data.py) dengan data buatan; satu tes membaca CSV arsip di repo."""

import pandas as pd

from src import muat_data

RSS_CONTOH = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>TurnBackHoax.id</title>
<item><title>[SALAH] Judul Satu</title><link>https://turnbackhoax.id/articles/37370-salah-judul-satu</link>
<pubDate>Wed, 05 Nov 2026 03:00:00 +0000</pubDate><category>Politik</category></item>
<item><title>[PENIPUAN] Judul Dua</title><link>https://turnbackhoax.id/articles/37369-penipuan-judul-dua</link>
<pubDate>Tue, 04 Nov 2026 03:00:00 +0000</pubDate></item>
</channel></rss>"""


def contoh_df():
    return pd.DataFrame({
        "id": [3, 2, 1, 4, 5],
        "tanggal": ["2026-11-01", "2025-06-15", "2020-01-20", "bukan tanggal", "6325-10-25"],
        "judul": ["[SALAH] Video Vaksin Mengandung Chip", "[PENIPUAN] Tautan Bansos", "(HOAX) Gempa Susulan Besar", "[SALAH] x", "[SALAH] y"],
        "label": ["SALAH", "PENIPUAN", "HOAX", "SALAH", "SALAH"],
        "kategori": ["Kesehatan", "", "other", "Bencana", ""],
        "ringkasan": ["Faktanya tidak", "", "", "", ""],
        "url": ["u3", "u2", "u1", "u4", "u5"],
    })


def test_kelompokkan_label():
    assert muat_data.kelompokkan_label("SALAH") == "Salah"
    assert muat_data.kelompokkan_label("HOAX") == "Salah"
    assert muat_data.kelompokkan_label("PENIPUAN") == "Penipuan"
    assert muat_data.kelompokkan_label("KLARIFIKASI") == "Klarifikasi"
    assert muat_data.kelompokkan_label("BENAR") == "Benar"
    assert muat_data.kelompokkan_label("BELUM TERBUKTI") == "Klarifikasi"
    assert muat_data.kelompokkan_label("PARODI") == "Satire"
    assert muat_data.kelompokkan_label("Top 5") == "Lainnya"
    assert muat_data.kelompokkan_label(None) == "Lainnya"


def test_rapikan_kategori():
    assert muat_data.rapikan_kategori("") == "Lainnya"
    assert muat_data.rapikan_kategori("other") == "Lainnya"
    assert muat_data.rapikan_kategori("politik") == "Politik"


def test_siapkan_menambah_kolom_turunan_dan_membuang_tanggal_rusak():
    df = muat_data.siapkan(contoh_df())
    assert len(df) == 3                                   # 'bukan tanggal' dan tahun 6325 dibuang
    assert list(df["id"]) == [3, 2, 1]                    # urut dari terbaru
    assert df.loc[0, "judul_bersih"] == "Video Vaksin Mengandung Chip"
    assert df.loc[0, "teks_tfidf"] == "vaksin mengandung chip"
    assert df.loc[0, "format"] == "Video"
    assert df.loc[1, "kategori"] == "Lainnya" and df.loc[2, "label_kelompok"] == "Salah"
    assert df.loc[2, "bulan"] == pd.Timestamp("2020-01-01") and df.loc[2, "tahun"] == 2020


def test_urai_rss():
    df = muat_data.urai_rss(RSS_CONTOH)
    assert len(df) == 2
    assert df.loc[0, "id"] == 37370 and df.loc[0, "kategori"] == "Politik"
    assert df.loc[0, "tanggal"] == pd.Timestamp("2026-11-05 10:00:00")  # 03.00 GMT = 10.00 WIB
    assert df.loc[1, "kategori"] == ""


def test_ambil_rss_gagal_tidak_melempar_error():
    hasil = muat_data.ambil_rss("http://127.0.0.1:9/tidak-ada", timeout=1)
    assert hasil.sumber == "gagal" and hasil.data.empty


def test_arsip_di_repo_terbaca():
    df = muat_data.muat_arsip()
    assert len(df) > 5000
    assert df["id"].is_unique
    assert df["tanggal"].min().year >= 2014 and df["tanggal"].max().year >= 2026
    assert {"Salah", "Penipuan"} <= set(df["label_kelompok"])
    assert df["teks_tfidf"].str.len().gt(0).mean() > 0.95
