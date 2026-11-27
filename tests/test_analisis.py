"""Uji mesin NLP (src/analisis.py) pada korpus kecil buatan: dua topik yang jelas terpisah."""

from types import SimpleNamespace

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


def test_beri_nama_bandingkan_kata_utuh():
    kk = [("polri", 0.5), ("ri", 0.4), ("air", 0.3), ("ai", 0.2)]
    assert analisis.beri_nama(kk, n=4) == "polri · ri · air · ai"


def test_ringkasan_klaster(df, model):
    r = analisis.ringkasan_klaster(df, model)
    assert len(r) == 2 and r["jumlah"].sum() == len(df)
    assert set(r.columns) >= {"nama", "jumlah", "persen", "kata_kunci", "kategori_dominan", "tahun_puncak", "contoh"}


def test_ringkasan_klaster_kategori_dan_tahun_kosong(df, model):
    kosong = df.assign(kategori=None, tahun=float("nan"))
    r = analisis.ringkasan_klaster(kosong, model)
    assert len(r) == 2
    assert (r["kategori_dominan"] == "-").all()
    assert r["tahun_puncak"].isna().all()


def test_tren_bulanan_mengisi_bulan_kosong(df):
    t = analisis.tren_bulanan(df, "kategori")
    assert t.index.min() == pd.Timestamp("2021-01-01") and t.index.max() == pd.Timestamp("2023-08-01")
    assert len(t) == 32 and int(t.values.sum()) == len(df)
    assert analisis.tren_bulanan(df[df["id"] < 0], "kategori").empty


def test_tren_bulanan_semua_tanpa_tanggal(df):
    tanpa = df.assign(tanggal=pd.NaT, bulan=pd.NaT)
    assert analisis.tren_bulanan(tanpa, "kategori").empty


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


def test_tren_kata_dan_frekuensi_tahan_teks_kosong(df):
    d = df.copy()
    d["teks_tfidf"] = d["teks_tfidf"].astype(object)
    baris_vaksin = d.index[d["teks_tfidf"].str.contains("vaksin")][0]
    d.loc[baris_vaksin, "teks_tfidf"] = None
    t = analisis.tren_kata(d, ["vaksin"])
    assert t["vaksin"].sum() == 7
    f = analisis.frekuensi_kata(d["teks_tfidf"], n=50)
    assert "nan" not in f and "None" not in f


def test_kata_naik_turun_tidak_tumpang_tindih():
    df = pd.DataFrame({
        "tanggal": pd.to_datetime(["2021-01-01"] * 4 + ["2023-01-01"] * 4),
        "teks_tfidf": ["vaksin chip", "vaksin chip", "vaksin", "banjir"] + ["gempa", "gempa banjir", "gempa", "chip"],
    })
    tab = analisis.kata_naik_turun(df, pd.Timestamp("2022-01-01"), min_total=1, n=15)
    naik = set(tab.loc[tab["arah"] == "Naik", "kata"])
    turun = set(tab.loc[tab["arah"] == "Turun", "kata"])
    assert naik.isdisjoint(turun)
    assert (tab.loc[tab["arah"] == "Naik", "rasio"] > 1).all()
    assert (tab.loc[tab["arah"] == "Turun", "rasio"] < 1).all()
    assert "gempa" in naik and "vaksin" in turun


def test_kata_naik_turun_tanpa_kata_lolos_ambang(df):
    tab = analisis.kata_naik_turun(df, pd.Timestamp("2022-01-01"), min_total=1000)
    assert tab.empty


def test_tren_bulanan_nilai_kosong_tetap_dihitung(df):
    d = df.copy()
    d.loc[d.index[:3], "format"] = None
    t = analisis.tren_bulanan(d, "format")
    assert int(t.values.sum()) == len(d)
    assert "Tidak diketahui" in t.columns


def test_tren_kata_data_kosong(df):
    assert analisis.tren_kata(df.iloc[0:0], ["vaksin"]).empty


def test_prediksi_klaim_tanpa_kata_dikenal_tidak_ditebak(model):
    hasil = analisis.prediksi_klaster(model, ["zzzqx wqqpl", "vaksin covid chip"])
    assert hasil[0] == -1
    assert hasil[1] != -1
    assert model.nama_topik(-1) == "(tidak dikenali)"
    assert model.nama_topik(hasil[1]) == model.nama[hasil[1]]


def test_cari_mirip_n_nol_atau_negatif_kosong(model):
    for n in (0, -1):
        idx, skor = analisis.cari_mirip(model, "vaksin covid chip", n=n)
        assert len(idx) == 0 and len(skor) == 0


def test_kata_kunci_klaster_batas_negatif_dan_bobot_seri():
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer().fit(["alpha beta gamma delta"])
    komponen = np.array([[0.5, 0.5, 0.5, 0.1]])   # alpha, beta, delta, gamma (urut abjad kosakata)
    assert analisis.kata_kunci_klaster(vec, komponen, n=-1) == {0: []}
    kata = [k for k, _ in analisis.kata_kunci_klaster(vec, komponen, n=3)[0]]
    assert kata == ["alpha", "beta", "delta"]


def test_kata_naik_turun_batas_negatif_kosong(df):
    assert analisis.kata_naik_turun(df, pd.Timestamp("2022-01-01"), min_total=3, n=-2).empty


def test_frekuensi_kata_seri_urut_abjad():
    kata = [f"kata{i:02d}" for i in range(40)]
    dokumen = pd.Series([" ".join(kata), " ".join(kata[::-1]), "zeta"])
    hasil = list(analisis.frekuensi_kata(dokumen, n=10))
    assert hasil == kata[:10]
    assert analisis.frekuensi_kata(dokumen, n=-1) == {}


def test_kata_dikenali_tanpa_duplikat(model):
    dikenal = analisis.kata_dikenali(model, "vaksin covid vaksin chip vaksin")
    assert dikenal.count("vaksin") == 1
    assert dikenal == list(dict.fromkeys(dikenal))


@pytest.mark.parametrize("n", [0, -2])
def test_beri_nama_n_nol_atau_negatif(n):
    kk = [("vaksin", 0.9), ("chip", 0.5), ("gempa", 0.3)]
    assert analisis.beri_nama(kk, n=n) == "(tanpa kata kunci)"


def test_ringkasan_klaster_tanpa_artikel(df, model):
    kosong = df.iloc[0:0]
    palsu = SimpleNamespace(label=np.array([], dtype=int), k=model.k, nama=model.nama, kata_kunci=model.kata_kunci)
    hasil = analisis.ringkasan_klaster(kosong, palsu)
    assert hasil.empty
    assert "jumlah" in hasil.columns


def test_ringkasan_klaster_ukuran_seri_urut_nomor():
    k = 40
    label = np.repeat(np.arange(k), [3 if i % 3 else 5 for i in range(k)])
    df = pd.DataFrame({"kategori": "x", "tahun": 2024, "label_kelompok": "Salah", "judul_bersih": "a"},
                      index=range(len(label)))
    palsu = SimpleNamespace(label=label, k=k, nama={i: f"t{i}" for i in range(k)}, kata_kunci={i: [] for i in range(k)})
    hasil = analisis.ringkasan_klaster(df, palsu)
    for _, grup in hasil.groupby("jumlah"):
        assert grup["klaster"].tolist() == sorted(grup["klaster"])
