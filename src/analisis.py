"""
src/analisis.py - Mesin NLP: TF-IDF, pengelompokan topik (NMF / K-Means), tren, kata naik, dan pencarian kemiripan
=====================================================================================================
Semua fungsi menerima/mengembalikan pandas atau numpy; tidak ada Streamlit di sini.

Alur utama:
    judul bersih (src/teks.py) -> TF-IDF (kata + pasangan kata) -> NMF (bawaan) atau K-Means (k topik)
    -> nama topik dari kata berbobot tertinggi di komponen / pusat klaster
    -> tren bulanan per klaster / kategori / label
    -> pencarian: cosine similarity antara klaim baru dan seluruh arsip
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from src import teks


# ----------------------------------------------------------------------------
# 1. TF-IDF
# ----------------------------------------------------------------------------
def bangun_tfidf(dokumen: list[str], min_df: int = 3, max_df: float = 0.5, max_fitur: int = 30000):
    """
    APA: ubah daftar judul bersih menjadi matriks TF-IDF (dokumen x kata).
    MENGAPA ngram (1,2): pasangan kata seperti "prabowo subianto" atau "kartu prakerja" lebih informatif
    daripada kata tunggalnya. min_df=3 membuang kata yang muncul di <3 judul (salah ketik, nama langka);
    max_df=0.5 membuang kata yang muncul di >50% judul (hampir pasti tidak membedakan topik).
    Teks sudah dibersihkan, sehingga tokenizer cukup str.split.
    """
    vec = TfidfVectorizer(
        tokenizer=str.split, preprocessor=None, lowercase=False, token_pattern=None,
        ngram_range=(1, 2), min_df=min_df, max_df=max_df, max_features=max_fitur, sublinear_tf=True,
    )
    X = vec.fit_transform(dokumen)
    return vec, X


# ----------------------------------------------------------------------------
# 2. Pengelompokan topik: NMF (bawaan) atau K-Means, plus penamaan klaster
# ----------------------------------------------------------------------------
METODE = ("NMF", "K-Means")


@dataclass
class ModelTopik:
    """
    Hasil pelatihan yang dipakai seluruh aplikasi, apa pun metodenya.

    - `label`     : nomor topik setiap dokumen (argmax bobot NMF, atau klaster K-Means)
    - `komponen`  : matriks (k x kata) yang menjadi dasar kata kunci: baris H untuk NMF, pusat klaster untuk K-Means
    - `model`     : objek sklearn (NMF atau KMeans) untuk memetakan dokumen baru
    """
    vectorizer: TfidfVectorizer
    X: sparse.csr_matrix
    model: object
    metode: str
    k: int
    label: np.ndarray
    komponen: np.ndarray
    kata_kunci: dict[int, list[tuple[str, float]]]
    nama: dict[int, str]

    def prediksi(self, V) -> np.ndarray:
        """Topik untuk vektor TF-IDF baru V (bentuk n x kata)."""
        if self.metode == "NMF":
            return self.model.transform(V).argmax(axis=1)
        return self.model.predict(V)


def klasterkan(X, k: int, seed: int = 42) -> KMeans:
    """K-Means pada vektor TF-IDF (sudah ternormalisasi L2, sehingga jarak Euclid ~ cosine)."""
    km = KMeans(n_clusters=k, n_init=5, random_state=seed, max_iter=300)
    km.fit(X)
    return km


def faktorkan(X, k: int, seed: int = 42) -> NMF:
    """
    APA: Non-negative Matrix Factorization: X (dokumen x kata) ≈ W (dokumen x topik) · H (topik x kata).
    MENGAPA bawaan: setiap dokumen mendapat bobot untuk semua topik, lalu diambil yang terbesar. Pada judul pendek
    hasilnya jauh lebih seimbang daripada K-Means, yang cenderung menumpuk 60-70% dokumen di satu klaster "sisa".
    init="nndsvda" membuat hasilnya deterministik dan cepat konvergen.
    """
    nmf = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=400)
    nmf.fit(X)
    return nmf


def kata_kunci_klaster(vec: TfidfVectorizer, komponen: np.ndarray, n: int = 10) -> dict[int, list[tuple[str, float]]]:
    """Kata dengan bobot terbesar di setiap baris `komponen` (pusat klaster / baris H) = ciri khas topik itu."""
    fitur = np.asarray(vec.get_feature_names_out())
    hasil = {}
    for i, baris in enumerate(komponen):
        urut = np.argsort(baris)[::-1][:n]
        hasil[i] = [(fitur[j], float(baris[j])) for j in urut if baris[j] > 0]
    return hasil


def beri_nama(kata_kunci: list[tuple[str, float]], n: int = 3) -> str:
    """Gabungkan kata kunci teratas yang tidak saling tumpang tindih: 'vaksin · covid · chip'."""
    terpilih: list[str] = []
    for kata, _ in kata_kunci:
        if any(kata in t or t in kata for t in terpilih):
            continue
        terpilih.append(kata)
        if len(terpilih) == n:
            break
    return " · ".join(terpilih) if terpilih else "(tanpa kata kunci)"


def rakit_model(vec: TfidfVectorizer, X, k: int, metode: str = "NMF", seed: int = 42) -> ModelTopik:
    """Latih NMF atau K-Means pada matriks TF-IDF yang sudah ada, lalu beri nama setiap topik."""
    if metode == "NMF":
        model = faktorkan(X, k, seed)
        label = model.transform(X).argmax(axis=1)
        komponen = model.components_
    elif metode == "K-Means":
        model = klasterkan(X, k, seed)
        label = model.labels_
        komponen = model.cluster_centers_
    else:
        raise ValueError(f"metode tidak dikenal: {metode}")
    kk = kata_kunci_klaster(vec, komponen)
    return ModelTopik(vec, X, model, metode, k, np.asarray(label), komponen, kk, {i: beri_nama(kk[i]) for i in kk})


def bangun_model(dokumen: list[str], k: int = 12, metode: str = "NMF", seed: int = 42) -> ModelTopik:
    vec, X = bangun_tfidf(dokumen)
    return rakit_model(vec, X, k, metode, seed)


def ringkasan_klaster(df: pd.DataFrame, model: ModelTopik) -> pd.DataFrame:
    """Satu baris per klaster: nama, ukuran, kata kunci, kategori dominan, tahun puncak, contoh judul."""
    d = df.assign(klaster=model.label)
    baris = []
    for i in range(model.k):
        sub = d[d["klaster"] == i]
        if sub.empty:
            continue
        kategori = sub["kategori"].value_counts()
        tahun = sub["tahun"].value_counts()
        baris.append({
            "klaster": i,
            "nama": model.nama[i],
            "jumlah": int(len(sub)),
            "persen": round(100 * len(sub) / len(d), 1),
            "kata_kunci": ", ".join(k for k, _ in model.kata_kunci[i][:8]),
            "kategori_dominan": f"{kategori.index[0]} ({100 * kategori.iloc[0] / len(sub):.0f}%)",
            "tahun_puncak": int(tahun.index[0]),
            "label_salah_persen": round(100 * (sub["label_kelompok"] == "Salah").mean(), 0),
            "contoh": sub["judul_bersih"].head(3).tolist(),
        })
    return pd.DataFrame(baris).sort_values("jumlah", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3. Tren waktu
# ----------------------------------------------------------------------------
def tren_bulanan(df: pd.DataFrame, kolom: str, mulai: pd.Timestamp | None = None) -> pd.DataFrame:
    """Tabel bulan x nilai `kolom` berisi jumlah artikel. Bulan tanpa artikel diisi 0."""
    d = df if mulai is None else df[df["tanggal"] >= mulai]
    if d.empty:
        return pd.DataFrame()
    pv = d.pivot_table(index="bulan", columns=kolom, values="id", aggfunc="count", fill_value=0)
    semua_bulan = pd.date_range(pv.index.min(), pv.index.max(), freq="MS")
    return pv.reindex(semua_bulan, fill_value=0).rename_axis("bulan")


def tren_kata(df: pd.DataFrame, kata: list[str]) -> pd.DataFrame:
    """Jumlah judul per bulan yang mengandung setiap kata (pencocokan kata utuh pada teks bersih)."""
    hasil = {}
    for k in kata:
        k_bersih = " ".join(teks.tokenisasi(k))
        if not k_bersih:          # kosong = hanya stopword/tanda baca, lewati
            continue
        cocok = df["teks_tfidf"].str.contains(rf"\b{k_bersih}\b", regex=True)
        hasil[k] = df.loc[cocok].groupby("bulan").size()
    if not hasil:
        return pd.DataFrame()
    out = pd.DataFrame(hasil).fillna(0).astype(int)
    semua_bulan = pd.date_range(df["bulan"].min(), df["bulan"].max(), freq="MS")
    return out.reindex(semua_bulan, fill_value=0).rename_axis("bulan")


def frekuensi_kata(dokumen: pd.Series, n: int = 150) -> dict[str, int]:
    """Hitung kemunculan kata tunggal pada kumpulan judul bersih (bahan word cloud dan grafik batang)."""
    c: Counter = Counter()
    for d in dokumen:
        c.update(set(str(d).split()))  # set: satu judul menyumbang maksimal 1 per kata
    return dict(c.most_common(n))


def kata_naik_turun(df: pd.DataFrame, batas: pd.Timestamp, min_total: int = 8, n: int = 15) -> pd.DataFrame:
    """
    Bandingkan seberapa sering sebuah kata muncul per 1.000 judul SEBELUM vs SESUDAH tanggal `batas`.
    rasio > 1 berarti kata itu makin ramai. Pemulusan +1 (Laplace) agar kata baru tidak memberi pembagian nol.
    """
    sebelum = df[df["tanggal"] < batas]["teks_tfidf"]
    sesudah = df[df["tanggal"] >= batas]["teks_tfidf"]
    if sebelum.empty or sesudah.empty:
        return pd.DataFrame()
    f_lama, f_baru = Counter(), Counter()
    for d in sebelum:
        f_lama.update(set(d.split()))
    for d in sesudah:
        f_baru.update(set(d.split()))
    kata = {k for k in set(f_lama) | set(f_baru) if f_lama[k] + f_baru[k] >= min_total}
    baris = []
    for k in kata:
        laju_lama = 1000 * (f_lama[k] + 1) / (len(sebelum) + 1)
        laju_baru = 1000 * (f_baru[k] + 1) / (len(sesudah) + 1)
        baris.append({"kata": k, "sebelum": f_lama[k], "sesudah": f_baru[k],
                      "per_1000_sebelum": round(laju_lama, 1), "per_1000_sesudah": round(laju_baru, 1),
                      "rasio": round(laju_baru / laju_lama, 2)})
    tab = pd.DataFrame(baris)
    naik = tab.sort_values("rasio", ascending=False).head(n).assign(arah="Naik")
    turun = tab.sort_values("rasio").head(n).assign(arah="Turun")
    return pd.concat([naik, turun], ignore_index=True)


# ----------------------------------------------------------------------------
# 4. Klaim baru: klaster terdekat dan hoaks serupa di arsip
# ----------------------------------------------------------------------------
def vektor_klaim(model: ModelTopik, klaim: list[str]):
    return model.vectorizer.transform([teks.bersihkan_judul(k) for k in klaim])


def prediksi_klaster(model: ModelTopik, klaim: list[str]) -> np.ndarray:
    """Topik terdekat untuk setiap klaim baru (judul bebas, boleh dengan awalan label)."""
    return model.prediksi(vektor_klaim(model, klaim))


def cari_mirip(model: ModelTopik, klaim: str, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Cosine similarity antara klaim dan semua judul arsip. Karena TF-IDF sudah L2-normal,
    cosine = perkalian titik. Mengembalikan (indeks baris, skor) terurut dari yang paling mirip.
    """
    v = vektor_klaim(model, [klaim])
    skor = np.asarray((model.X @ v.T).todense()).ravel()
    urut = np.argsort(skor)[::-1][:n]
    return urut, skor[urut]
