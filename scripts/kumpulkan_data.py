"""
scripts/kumpulkan_data.py - Mengumpulkan arsip artikel cek fakta Turnbackhoax (Mafindo)
=======================================================================================
Jalankan dari akar repo:

    python scripts/kumpulkan_data.py            # perbarui: ambil artikel baru sampai bertemu yang sudah ada
    python scripts/kumpulkan_data.py --semua    # unduh ulang seluruh arsip (±1.500 halaman, server ±5 s/halaman, ±45 menit)

Sumber: halaman daftar publik https://turnbackhoax.id/articles?page=N (10 artikel per halaman).
Setiap kartu memuat id, judul, ringkasan singkat, kategori, dan tanggal terbit. Isi artikel tidak diunduh;
yang disimpan hanya metadata yang memang ditampilkan di halaman daftar.

Hasil: data/hoaks_turnbackhoax.csv dengan kolom
    id, tanggal, judul, label, kategori, ringkasan, url
Baris dengan tanggal mustahil (tahun 1000, 6325, masa depan) atau tanpa tanggal dibuang saat disimpan.

Etika pengambilan: permintaan dibatasi beberapa utas dengan jeda, mematuhi kode 429/5xx dengan pengulangan
bertahap, dan hanya mengakses halaman yang memang disediakan untuk pengunjung.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

URL_DAFTAR = "https://turnbackhoax.id/articles"
BERKAS_KELUARAN = Path(__file__).resolve().parent.parent / "data" / "hoaks_turnbackhoax.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; disinfo-radar/1.0; +https://github.com/Avalon2222/disinfo-radar)"}
KOLOM = ["id", "tanggal", "judul", "label", "kategori", "ringkasan", "url"]

POLA_KARTU = re.compile(r'<div class="news-card-h-alt.*?</div>\s*</div>\s*</div>', re.S)
POLA_URL = re.compile(r'href="(https://turnbackhoax\.id/articles/(\d+)-[^"]*)"')
POLA_JUDUL = re.compile(r'<img[^>]*alt="([^"]*)"')
POLA_RINGKASAN = re.compile(r'<span class="hidden xl:block">(.*?)</span>', re.S)
POLA_KATEGORI = re.compile(r'category=([^&"]+)&page=1">([^<]*)<')
POLA_TANGGAL = re.compile(r'(\d{2})/(\d{2})/(\d{4})')
POLA_LABEL = re.compile(r'^\s*[\[\(]\s*([^\]\)]+?)\s*[\]\)]')


def bersihkan(teks: str) -> str:
    teks = html.unescape(re.sub(r"<[^>]+>", " ", teks))
    return re.sub(r"\s+", " ", teks).strip()


def tebak_label(judul: str) -> str:
    """Label cek fakta diambil dari awalan judul, misalnya '[SALAH] ...' -> 'SALAH'."""
    m = POLA_LABEL.match(judul)
    return m.group(1).strip().upper() if m else "TANPA LABEL"


def urai_halaman(teks_html: str) -> list[dict]:
    baris = []
    for kartu in POLA_KARTU.findall(teks_html):
        m_url = POLA_URL.search(kartu)
        if not m_url:
            continue
        judul = bersihkan(POLA_JUDUL.search(kartu).group(1)) if POLA_JUDUL.search(kartu) else ""
        m_ring = POLA_RINGKASAN.search(kartu)
        m_kat = POLA_KATEGORI.search(kartu)
        m_tgl = POLA_TANGGAL.search(kartu)
        tanggal = f"{m_tgl.group(3)}-{m_tgl.group(2)}-{m_tgl.group(1)}" if m_tgl else ""
        baris.append({
            "id": int(m_url.group(2)),
            "tanggal": tanggal,
            "judul": judul,
            "label": tebak_label(judul),
            "kategori": bersihkan(m_kat.group(2)) if m_kat else "",
            "ringkasan": bersihkan(m_ring.group(1)) if m_ring else "",
            "url": m_url.group(1),
        })
    return baris


def ambil_halaman(halaman: int, sesi: requests.Session, percobaan: int = 5) -> list[dict]:
    for ke in range(percobaan):
        try:
            r = sesi.get(URL_DAFTAR, params={"page": halaman}, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return urai_halaman(r.text)
            if r.status_code == 404:
                return []
        except requests.RequestException:
            pass
        time.sleep(1.5 * (ke + 1))
    print(f"  ! halaman {halaman} gagal setelah {percobaan} percobaan", file=sys.stderr)
    return []


def muat_lama() -> pd.DataFrame:
    if BERKAS_KELUARAN.exists():
        return pd.read_csv(BERKAS_KELUARAN)
    return pd.DataFrame(columns=KOLOM)


TANGGAL_MIN = "2014-01-01"   # artikel tertua yang masuk akal


def buang_tanggal_mustahil(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sebagian kecil kartu (±6%) tampil dengan tanggal mustahil di situs, misalnya tahun 1000 atau 6325,
    atau tanpa tanggal sama sekali. Baris seperti itu tidak bisa dipakai untuk tren dan dibuang di sini.
    """
    tgl = pd.to_datetime(df["tanggal"], errors="coerce", format="%Y-%m-%d")
    batas_akhir = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    return df[(tgl >= pd.Timestamp(TANGGAL_MIN)) & (tgl <= batas_akhir)]


def simpan(df: pd.DataFrame, senyap: bool = False) -> None:
    df = buang_tanggal_mustahil(df.drop_duplicates("id")).sort_values(["tanggal", "id"], ascending=False)
    df = df[KOLOM]
    BERKAS_KELUARAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BERKAS_KELUARAN, index=False)
    if not senyap:
        print(f"Tersimpan {len(df):,} artikel ke {BERKAS_KELUARAN}")


def unduh_semua(mulai: int = 1, utas: int = 3) -> pd.DataFrame:
    """
    Unduh halaman mulai..∞ secara paralel per blok, berhenti setelah satu blok penuh kosong.
    Hasil disimpan ke CSV setiap blok (digabung dengan yang sudah ada), sehingga proses yang terputus
    bisa dilanjutkan dengan --mulai <halaman terakhir>.
    """
    semua: list[dict] = muat_lama().to_dict("records")
    halaman = mulai
    sesi = requests.Session()
    with ThreadPoolExecutor(max_workers=utas) as pool:
        while True:
            blok = list(range(halaman, halaman + utas * 4))
            hasil = list(pool.map(lambda h: ambil_halaman(h, sesi), blok))
            n_baru = sum(len(x) for x in hasil)
            for x in hasil:
                semua.extend(x)
            if n_baru:
                simpan(pd.DataFrame(semua, columns=KOLOM), senyap=True)
            print(f"halaman {blok[0]}-{blok[-1]}: {n_baru} artikel (total {len(semua):,})", flush=True)
            if n_baru == 0:
                break
            halaman = blok[-1] + 1
            time.sleep(0.5)
    return pd.DataFrame(semua, columns=KOLOM)


def perbarui(utas: int = 3, batas_halaman: int = 400) -> pd.DataFrame:
    """Ambil halaman terbaru satu per satu sampai seluruh halaman berisi artikel yang sudah tersimpan."""
    lama = muat_lama()
    dikenal = set(lama["id"].astype(int)) if len(lama) else set()
    baru: list[dict] = []
    sesi = requests.Session()
    for halaman in range(1, batas_halaman + 1):
        kartu = ambil_halaman(halaman, sesi)
        if not kartu:
            break
        segar = [k for k in kartu if k["id"] not in dikenal]
        baru.extend(segar)
        print(f"halaman {halaman}: {len(segar)} artikel baru", flush=True)
        if not segar and dikenal:
            break
        time.sleep(0.3)
    gabungan = pd.concat([pd.DataFrame(baru, columns=KOLOM), lama], ignore_index=True)
    print(f"{len(baru)} artikel baru ditambahkan")
    return gabungan


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--semua", action="store_true", help="unduh ulang seluruh arsip dari halaman 1")
    ap.add_argument("--mulai", type=int, default=1, help="halaman awal (hanya bersama --semua)")
    ap.add_argument("--utas", type=int, default=3, help="jumlah permintaan paralel")
    arg = ap.parse_args()
    df_hasil = unduh_semua(arg.mulai, arg.utas) if arg.semua else perbarui(arg.utas)
    simpan(df_hasil)
