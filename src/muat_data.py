"""
src/muat_data.py - Memuat arsip cek fakta dan mengambil artikel terbaru
=======================================================================
Dua sumber:
  1. data/hoaks_turnbackhoax.csv  : arsip metadata artikel Turnbackhoax (dibangun scripts/kumpulkan_data.py)
  2. https://turnbackhoax.id/feed : RSS resmi berisi ±10 artikel terakhir, dipakai untuk "radar live"

Fungsi di sini hanya membaca dan merapikan; tidak ada Streamlit supaya bisa diuji langsung dengan pytest.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

from src import teks

DIR_DATA = Path(__file__).resolve().parent.parent / "data"
BERKAS_ARSIP = DIR_DATA / "hoaks_turnbackhoax.csv"
URL_RSS = "https://turnbackhoax.id/feed"
WIB = timezone(timedelta(hours=7), "WIB")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; disinfo-radar/1.0)"}

# Label di judul Turnbackhoax berubah gaya dari tahun ke tahun; disatukan ke beberapa kelompok.
PETA_LABEL = {
    "SALAH": "Salah", "HOAX": "Salah", "HOAKS": "Salah", "FALSE": "Salah", "FAKE": "Salah", "FITNAH": "Salah",
    "DISINFORMASI": "Salah", "MISINFORMASI": "Salah", "MISLEADING": "Salah", "PALSU": "Salah", "HASUT": "Salah",
    "PENIPUAN": "Penipuan", "SCAM": "Penipuan", "PHISHING": "Penipuan",
    "BENAR": "Benar", "TRUE": "Benar", "FAKTA": "Benar",
    "KLARIFIKASI": "Klarifikasi", "CLARIFICATION": "Klarifikasi", "SEBAGIAN BENAR": "Klarifikasi",
    "SEBAGIAN SALAH": "Klarifikasi", "MENYESATKAN": "Klarifikasi", "BELUM TERBUKTI": "Klarifikasi",
    "SATIRE": "Satire", "SATIR": "Satire", "PARODI": "Satire", "KOMEDI": "Satire",
}
TAHUN_MIN = 2014   # artikel tertua yang masuk akal di arsip
URUTAN_LABEL = ["Salah", "Penipuan", "Klarifikasi", "Benar", "Satire", "Lainnya"]
WARNA_LABEL = {
    "Salah": "#ff5c8a", "Penipuan": "#ffb000", "Klarifikasi": "#38bdf8",
    "Benar": "#4ade80", "Satire": "#c084fc", "Lainnya": "#8b93a7",
}
WARNA_UTAMA = "#a78bfa"


def _teks(v) -> str:
    """Nilai sel sebagai teks; kosong untuk None/NaN/pd.NA.

    Kolom dibaca dengan dtype "string", jadi sel kosong berupa pd.NA, dan `pd.NA or ""` melempar TypeError.
    """
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ""
    return str(v)


def kelompokkan_label(label: str) -> str:
    """'SALAH' -> 'Salah', 'HOAX' -> 'Salah', 'PENIPUAN' -> 'Penipuan', sisanya 'Lainnya'."""
    kunci = re.sub(r"[^A-Z ]", "", _teks(label).upper()).strip()
    if kunci in PETA_LABEL:
        return PETA_LABEL[kunci]
    for k, v in PETA_LABEL.items():
        if kunci.startswith(k):
            return v
    return "Lainnya"


def rapikan_kategori(kategori: str) -> str:
    """Kategori kosong atau 'Other' -> 'Lainnya'; kapitalisasi disamakan."""
    k = _teks(kategori).strip()
    if not k or k.lower() in {"other", "uncategorized", "lainnya", "nan"}:
        return "Lainnya"
    # "KESEHATAN", "kesehatan", dan "Kesehatan" harus jadi satu kategori, bukan tiga.
    return k[:1].upper() + k[1:].lower()


def siapkan(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan kolom turunan yang dipakai seluruh aplikasi."""
    df = df.copy()
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    # id dipakai sebagai kunci artikel; baris tanpa id angka tidak bisa dipakai dan dibuang.
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    # "inf" lolos dari dropna lalu membuat astype(int) di bawah error untuk seluruh arsip.
    df["id"] = df["id"].where(df["id"].abs() != float("inf"))
    df = df.dropna(subset=["id", "tanggal", "judul"]).copy()
    # Sebagian kecil artikel tampil dengan tanggal mustahil di situs (tahun 1000, 6325, atau di masa depan); dibuang.
    batas_akhir = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    df = df[(df["tanggal"].dt.year >= TAHUN_MIN) & (df["tanggal"] <= batas_akhir)].copy()
    df["id"] = df["id"].astype(int)
    # id adalah kunci artikel: artikel yang terambil dua kali (mis. saat pengumpul data diulang)
    # cukup dihitung sekali, pakai versi dengan tanggal terbaru.
    # mergesort: salinan yang terambil belakangan (lebih bawah di CSV) menang meski tanggalnya sama.
    df = df.sort_values("tanggal", kind="mergesort").drop_duplicates("id", keep="last")
    # Pengumpul data memberi "TANPA LABEL" bila kurung label tidak lengkap ("SALAH] ..."); ambil dari judul.
    tanpa_label = df["label"].map(_teks).str.strip().str.upper().isin(["", "TANPA LABEL"])
    dari_judul = df.loc[tanpa_label, "judul"].map(teks.label_dari_judul)
    df.loc[dari_judul[dari_judul != ""].index, "label"] = dari_judul[dari_judul != ""]
    df["label_kelompok"] = df["label"].map(kelompokkan_label)
    df["kategori"] = df["kategori"].map(rapikan_kategori)
    df["judul_bersih"] = df["judul"].map(teks.buang_label)
    df["teks_tfidf"] = df["judul"].map(teks.bersihkan_judul)
    df["format"] = df["judul"].map(teks.deteksi_format)
    df["bulan"] = df["tanggal"].dt.to_period("M").dt.to_timestamp()
    df["tahun"] = df["tanggal"].dt.year
    # Banyak artikel terbit di hari yang sama; id (nomor artikel di situs) memutus seri agar
    # daftar "terbaru" tidak berganti urutan setiap kali data dimuat ulang.
    return df.sort_values(["tanggal", "id"], ascending=False).reset_index(drop=True)


def muat_arsip(path: Path = BERKAS_ARSIP) -> pd.DataFrame:
    """Baca CSV arsip dan siapkan kolom turunan."""
    df = pd.read_csv(path, dtype={"ringkasan": "string", "kategori": "string", "label": "string"})
    df["ringkasan"] = df["ringkasan"].fillna("")
    return siapkan(df)


@dataclass
class HasilRSS:
    data: pd.DataFrame
    sumber: str      # "live" atau "gagal"
    pesan: str = ""


def _ke_wib(waktu: datetime) -> datetime:
    """Samakan zona waktu dengan arsip (tanggal WIB), lalu buang info zona.

    pubDate RSS biasanya dalam GMT. Tanpa konversi, artikel yang terbit pukul 05.00 WIB
    akan tercatat di tanggal sehari sebelumnya.
    """
    if waktu.tzinfo is None:
        return waktu
    return waktu.astimezone(WIB).replace(tzinfo=None)


def urai_rss(teks_xml: str) -> pd.DataFrame:
    """Ubah XML RSS menjadi tabel id, tanggal, judul, url, kategori."""
    akar = ET.fromstring(teks_xml)
    baris = []
    for item in akar.iter("item"):
        judul = (item.findtext("title") or "").strip()
        url = (item.findtext("link") or "").strip()
        tgl = item.findtext("pubDate")
        m = re.search(r"/articles/(\d+)", url)
        try:
            tanggal = _ke_wib(parsedate_to_datetime(tgl)) if tgl else pd.NaT
        except (TypeError, ValueError):
            tanggal = pd.NaT
        baris.append({
            "id": int(m.group(1)) if m else -1,
            "tanggal": tanggal,
            "judul": judul,
            "url": url,
            "kategori": (item.findtext("category") or "").strip(),
        })
    df = pd.DataFrame(baris, columns=["id", "tanggal", "judul", "url", "kategori"])
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    return df


def ambil_rss(url: str = URL_RSS, timeout: int = 12) -> HasilRSS:
    """Unduh RSS terbaru. Bila gagal (tanpa internet, server sibuk), kembalikan tabel kosong + pesan."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        df = urai_rss(r.text)
        if df.empty:
            return HasilRSS(df, "gagal", "RSS kosong")
        return HasilRSS(df, "live")
    except (requests.RequestException, ET.ParseError) as e:
        return HasilRSS(pd.DataFrame(columns=["id", "tanggal", "judul", "url", "kategori"]), "gagal", str(e)[:120])
