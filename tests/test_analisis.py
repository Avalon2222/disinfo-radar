"""Uji mesin NLP (src/analisis.py) pada korpus kecil buatan: dua topik yang jelas terpisah."""

import numpy as np
import pandas as pd
import pytest

from src import analisis, muat_data

JUDUL_VAKSIN = [
    "[SALAH] Vaksin Covid Mengandung Chip Pelacak", "[SALAH] Vaksin Covid Menyebabkan Kemandulan",
    "[SALAH] Vaksin Sinovac Berisi Microchip", "[SALAH] Efek Samping Vaksin Covid Mematikan",
    "[SALAH] Vaksin Covid Mengubah DNA Manusia", "[SALAH] Chip Vaksin Dikendalikan Sinyal 5G",
    "[SALAH] Vaksin Covid Haram dan Berbahaya", "[SALAH] Jutaan Orang Tewas Akibat Vaksin Covid",
]
JUDUL_GEMPA = [
    "[SALAH] Gempa Susulan Magnitudo 9 Akan Guncang Jakarta", "[SALAH] BMKG Prediksi Gempa Besar Malam Ini",
    "[SALAH] Tsunami Setinggi 30 Meter Setelah Gempa Cianjur", "[SALAH] Gempa Megathrust Pasti Terjadi Pekan Ini",
    "[SALAH] Video Gempa Turki Diklaim Gempa Cianjur", "[SALAH] BMKG Umumkan Tsunami Akibat Gempa Susulan",
    "[SALAH] Gempa Magnitudo 8 Guncang Bali Hari Ini", "[SALAH] Peringatan Tsunami BMKG Untuk Pantai Selatan",
]


@pytest.fixture(scope="module")
def df():
    n = len(JUDUL_VAKSIN) + len(JUDUL_GEMPA)
    tanggal = ["2021-0%d-10" % (i % 9 + 1) for i in range(len(JUDUL_VAKSIN))] + ["2023-%02d-10" % (i % 12 + 1) for i in range(len(JUDUL_GEMPA))]
    d = pd.DataFrame({
        "id": range(n), "tanggal": tanggal, "judul": JUDUL_VAKSIN + JUDUL_GEMPA,
        "label": ["SALAH"] * n, "kategori": ["Kesehatan"] * len(JUDUL_VAKSIN) + ["Bencana"] * len(JUDUL_GEMPA),
        "ringkasan": [""] * n, "url": [f"u{i}" for i in range(n)],
    })
    return muat_data.siapkan(d)


@pytest.fixture(scope="module", params=["NMF", "K-Means"])
def model(request, df):
    """bangun_model memakai min_df=3 yang terlalu ketat untuk 16 judul; rakit manual dengan min_df=1."""
    vec, X = analisis.bangun_tfidf(df["teks_tfidf"].tolist(), min_df=1, max_df=0.9)
    return analisis.rakit_model(vec, X, 2, request.param, seed=0)


def test_tfidf_bentuk_matriks(df):
    vec, X = analisis.bangun_tfidf(df["teks_tfidf"].tolist(), min_df=1, max_df=0.9)
    assert X.shape[0] == len(df)
    assert "vaksin" in vec.vocabulary_ and "gempa" in vec.vocabulary_
    assert np.allclose(np.asarray(X.multiply(X).sum(axis=1)).ravel(), 1.0)   # baris ternormalisasi L2


def test_metode_tidak_dikenal():
    vec, X = analisis.bangun_tfidf(["vaksin covid", "gempa bumi"], min_df=1, max_df=1.0)
    with pytest.raises(ValueError):
        analisis.rakit_model(vec, X, 2, "LDA")


def test_model_memisahkan_dua_topik(df, model):
    d = df.assign(klaster=model.label)
    klaster_vaksin = d[d["kategori"] == "Kesehatan"]["klaster"].unique()
    klaster_gempa = d[d["kategori"] == "Bencana"]["klaster"].unique()
    assert len(klaster_vaksin) == 1 and len(klaster_gempa) == 1 and klaster_vaksin[0] != klaster_gempa[0]


def test_nama_klaster_memuat_kata_kunci(model):
    nama = " ".join(model.nama.values())
    assert "vaksin" in nama and "gempa" in nama


def test_beri_nama_menghindari_tumpang_tindih():
    kk = [("vaksin covid", 0.5), ("vaksin", 0.4), ("covid", 0.3), ("chip", 0.2), ("gempa", 0.1)]
    assert analisis.beri_nama(kk) == "vaksin covid · chip · gempa"
    assert analisis.beri_nama([]) == "(tanpa kata kunci)"


def test_ringkasan_klaster(df, model):
    r = analisis.ringkasan_klaster(df, model)
    assert len(r) == 2 and r["jumlah"].sum() == len(df)
    assert set(r.columns) >= {"nama", "jumlah", "persen", "kata_kunci", "kategori_dominan", "tahun_puncak", "contoh"}


def test_tren_bulanan_mengisi_bulan_kosong(df):
    t = analisis.tren_bulanan(df, "kategori")
    assert t.index.min() == pd.Timestamp("2021-01-01") and t.index.max() == pd.Timestamp("2023-08-01")
    assert len(t) == 32 and int(t.values.sum()) == len(df)
    assert analisis.tren_bulanan(df[df["id"] < 0], "kategori").empty


def test_tren_kata(df):
    t = analisis.tren_kata(df, ["vaksin", "Gempa", "yang"])   # 'yang' = stopword -> dilewati
    assert list(t.columns) == ["vaksin", "Gempa"]
    assert t["vaksin"].sum() == 8 and t["Gempa"].sum() == 7


def test_frekuensi_kata(df):
    f = analisis.frekuensi_kata(df["teks_tfidf"], n=5)
    assert f["vaksin"] == 8 and f["gempa"] == 7 and len(f) == 5   # satu judul gempa hanya menyebut tsunami


def test_kata_naik_turun(df):
    tab = analisis.kata_naik_turun(df, pd.Timestamp("2022-01-01"), min_total=3, n=3)
    naik = tab[tab["arah"] == "Naik"]["kata"].tolist()
    turun = tab[tab["arah"] == "Turun"]["kata"].tolist()
    assert "gempa" in naik and "vaksin" in turun
    assert analisis.kata_naik_turun(df, pd.Timestamp("2010-01-01")).empty


def test_prediksi_dan_cari_mirip(df, model):
    d = df.assign(klaster=model.label)
    k_gempa = d[d["kategori"] == "Bencana"]["klaster"].iloc[0]
    assert analisis.prediksi_klaster(model, ["BMKG: gempa besar akan terjadi besok"])[0] == k_gempa
    idx, skor = analisis.cari_mirip(model, "vaksin covid chip 5G", n=3)
    assert len(idx) == 3 and skor[0] >= skor[1] >= skor[2] > 0
    assert all(df.loc[idx, "kategori"] == "Kesehatan")


def test_cari_mirip_buang_skor_nol(df, model):
    idx, skor = analisis.cari_mirip(model, "vaksin covid chip 5G", n=len(df))
    assert len(idx) < len(df)                       # judul gempa tidak berbagi kata, jadi tidak ikut
    assert (skor > 0).all()
    assert all(df.loc[idx, "kategori"] == "Kesehatan")


def test_cari_mirip_klaim_asing_hasil_kosong(model):
    idx, skor = analisis.cari_mirip(model, "resep rendang padang", n=5)
    assert len(idx) == 0 and len(skor) == 0
    assert analisis.kata_dikenali(model, "resep rendang padang") == []


def test_kata_dikenali_pakai_pembersih_yang_sama(model):
    dikenal = analisis.kata_dikenali(model, "[SALAH] Vaksin COVID mengandung chip!")
    assert "vaksin" in dikenal and "covid" in dikenal
    assert "salah" not in dikenal                   # label di depan judul dibuang dulu


def test_kata_kunci_klaster_urut_dan_buang_bobot_nol():
    vec, _ = analisis.bangun_tfidf(["vaksin covid chip", "gempa tsunami bmkg"], min_df=1, max_df=1.0)
    fitur = list(vec.get_feature_names_out())
    komponen = np.zeros((2, len(fitur)))
    komponen[0, fitur.index("vaksin")] = 0.9
    komponen[0, fitur.index("chip")] = 0.4
    komponen[1, fitur.index("gempa")] = 0.7
    kk = analisis.kata_kunci_klaster(vec, komponen, n=5)
    assert [k for k, _ in kk[0]] == ["vaksin", "chip"]     # terurut menurun, kata berbobot 0 dibuang
    assert [k for k, _ in kk[1]] == ["gempa"]
    assert all(isinstance(b, float) for _, b in kk[0])


def test_kata_kunci_klaster_dibatasi_n(model):
    kk = analisis.kata_kunci_klaster(model.vectorizer, np.ones((1, len(model.vectorizer.vocabulary_))), n=3)
    assert len(kk[0]) == 3


def test_vektor_klaim_membersihkan_label_dulu(model):
    # Awalan label "[SALAH]" tidak boleh mengubah vektor: klaim dibersihkan dengan cara yang sama seperti arsip.
    v = analisis.vektor_klaim(model, ["[SALAH] Vaksin Covid Mengandung Chip", "Vaksin Covid Mengandung Chip"])
    assert v.shape[0] == 2
    assert np.allclose(v[0].toarray(), v[1].toarray())


def _matriks_dua_topik():
    from sklearn.feature_extraction.text import TfidfVectorizer

    teks = ["vaksin covid chip"] * 4 + ["pemilu suara curang"] * 4
    return TfidfVectorizer().fit_transform(teks)


def test_klasterkan_deterministik_dan_pisah_topik():
    X = _matriks_dua_topik()
    a, b = analisis.klasterkan(X, 2), analisis.klasterkan(X, 2)
    assert (a.labels_ == b.labels_).all()                       # seed tetap -> hasil sama
    assert len(set(a.labels_[:4])) == 1 and a.labels_[0] != a.labels_[4]
    assert a.cluster_centers_.shape == (2, X.shape[1])


def test_faktorkan_bobot_tidak_negatif():
    X = _matriks_dua_topik()
    nmf = analisis.faktorkan(X, 2)
    assert nmf.components_.shape == (2, X.shape[1])
    assert (nmf.components_ >= 0).all()
    W = nmf.transform(X)
    assert W[:4].argmax(axis=1).tolist() == [W[0].argmax()] * 4
    assert W[0].argmax() != W[4].argmax()
