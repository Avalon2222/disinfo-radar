"""
app.py - Disinfo Radar
======================
Jalankan dengan:   streamlit run app.py

Alur besar file ini (baca dari atas ke bawah):
  1. Import library + konfigurasi halaman
  2. Muat arsip cek fakta Turnbackhoax (src/muat_data.py) dan siapkan kolom turunan
  3. Sidebar: rentang tahun, kategori, jumlah klaster, learning mode
  4. Bangun model TF-IDF + NMF (atau K-Means) pada data terpilih (src/analisis.py), di-cache
  5. Tampilkan: angka ringkasan -> tab Radar / Topic clusters / Keyword trends / Similarity check /
     Modus / Data & Sources / How it works

Setiap bagian diberi komentar "APA" dan "MENGAPA". Penjelasan istilah ada di docs/BELAJAR.md.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

from src import analisis, muat_data, teks
from src.muat_data import URUTAN_LABEL, WARNA_LABEL, WARNA_UTAMA

# ----------------------------------------------------------------------------
# 0. KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Disinfo Radar", page_icon="📡", layout="wide")

TEMPLATE = "plotly_dark"
PALET = ["#a78bfa", "#ff5c8a", "#ffb000", "#38bdf8", "#4ade80", "#f472b6", "#fb923c", "#2dd4bf",
         "#c084fc", "#facc15", "#60a5fa", "#f87171", "#a3e635", "#e879f9", "#22d3ee", "#fbbf24",
         "#818cf8", "#34d399", "#f9a8d4", "#94a3b8"]
DISCLAIMER = (
    "ℹ️ Seluruh data berasal dari **arsip publik** artikel cek fakta Turnbackhoax (Mafindo): judul, label, kategori, "
    "dan tanggal terbit yang memang ditampilkan di halaman daftar artikelnya. Aplikasi ini mengelompokkan dan "
    "menghitung, bukan memutuskan benar atau salah; penilaian tetap milik pemeriksa fakta. Proyek edukatif dan "
    "portofolio pribadi, tidak berafiliasi dengan Mafindo atau instansi mana pun."
)
WARNA_FORMAT = {"Video": "#ff5c8a", "Foto / gambar": "#ffb000", "Tautan / situs": "#38bdf8",
                "Pesan berantai": "#4ade80", "Teks / lainnya": "#8b93a7"}


def rapikan(fig: go.Figure, tinggi: int = 420) -> go.Figure:
    """Gaya seragam untuk semua grafik Plotly: latar gelap, grid tipis, legenda di atas."""
    fig.update_layout(
        template=TEMPLATE, height=tinggi,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=20, t=95, b=10),
        title=dict(y=0.99, yanchor="top", x=0, xanchor="left", font=dict(size=15)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None),
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#2a3242", zeroline=False)
    return fig


def tautan_judul(df: pd.DataFrame) -> pd.DataFrame:
    """Kolom tampilan standar untuk tabel artikel: tanggal, label, kategori, judul, URL."""
    return pd.DataFrame({
        "Tanggal": df["tanggal"].dt.strftime("%d/%m/%Y"),
        "Label": df["label_kelompok"],
        "Kategori": df["kategori"],
        "Judul": df["judul_bersih"],
        "URL": df["url"],
    })


KONFIG_TABEL = {"URL": st.column_config.LinkColumn("Artikel", display_text="buka ↗")}


# ----------------------------------------------------------------------------
# 1. MUAT DATA
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Membaca arsip cek fakta...")
def muat_arsip() -> pd.DataFrame:
    return muat_data.muat_arsip()


@st.cache_data(ttl=1800, show_spinner=False)
def ambil_rss(_waktu_kunci: str):
    """RSS Turnbackhoax (±10 artikel terakhir). Kunci waktu per 30 menit agar cache kedaluwarsa teratur."""
    return muat_data.ambil_rss()


@st.cache_resource(show_spinner="Melatih TF-IDF dan model topik...")
def bangun_model(ids: tuple[int, ...], dokumen: tuple[str, ...], k: int, metode: str) -> analisis.ModelTopik:
    """
    APA: TF-IDF + NMF (atau K-Means) pada judul yang sedang terpilih (filter tahun/kategori).
    MENGAPA di-cache dengan kunci (ids, k, metode): model hanya dilatih ulang bila filter, k, atau metode berubah.
    """
    return analisis.bangun_model(list(dokumen), k=k, metode=metode)


df_arsip = muat_arsip()
TANGGAL_SNAPSHOT = df_arsip["tanggal"].max()

# ----------------------------------------------------------------------------
# 2. SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("📡 Disinfo Radar")
    st.caption("Pola hoaks Indonesia dari ribuan artikel cek fakta, dikelompokkan otomatis dengan NLP.")

    st.subheader("Filter")
    thn_min, thn_maks = int(df_arsip["tahun"].min()), int(df_arsip["tahun"].max())
    if thn_min < thn_maks:
        rentang = st.slider("Rentang tahun", thn_min, thn_maks, (max(thn_min, thn_maks - 6), thn_maks))
    else:
        rentang = (thn_min, thn_maks)
    semua_kategori = df_arsip["kategori"].value_counts().index.tolist()
    kategori_pilih = st.multiselect("Kategori", semua_kategori, default=semua_kategori,
                                    help="Kategori yang diberikan Turnbackhoax pada setiap artikel.")

    st.subheader("Model")
    k_klaster = st.slider("Jumlah topik", 4, 24, 12,
                          help="Terlalu kecil: topik bercampur. Terlalu besar: topik terpecah-pecah. 10-14 biasanya pas.")
    metode = st.radio("Metode", analisis.METODE, horizontal=True,
                      help="NMF: setiap judul diberi bobot untuk semua topik, diambil yang terbesar; hasilnya seimbang. "
                           "K-Means: klasterisasi klasik; pada judul pendek cenderung membuat satu klaster 'sisa' yang besar.")

    mode_belajar = st.toggle("🎓 Learning mode", value=False, help="Tampilkan rumus dan potongan kode di setiap bagian.")
    st.caption(f"Arsip: {len(df_arsip):,} artikel, {df_arsip['tanggal'].min():%d %b %Y} – {TANGGAL_SNAPSHOT:%d %b %Y}. "
               "Perbarui dengan `python scripts/kumpulkan_data.py`.")

df = df_arsip[(df_arsip["tahun"].between(*rentang)) & (df_arsip["kategori"].isin(kategori_pilih))].reset_index(drop=True)
if len(df) < 50:
    st.warning("Data terpilih terlalu sedikit untuk dikelompokkan (minimal 50 artikel). Longgarkan filter di sidebar.")
    st.stop()

model = bangun_model(tuple(df["id"]), tuple(df["teks_tfidf"]), k_klaster, metode)
df["klaster"] = model.label
df["topik"] = df["klaster"].map(model.nama)
ringkas = analisis.ringkasan_klaster(df, model)
URUTAN_TOPIK = ringkas["nama"].tolist()
WARNA_TOPIK = {nama: PALET[i % len(PALET)] for i, nama in enumerate(URUTAN_TOPIK)}

# ----------------------------------------------------------------------------
# 3. HEADER + ANGKA RINGKASAN
# ----------------------------------------------------------------------------
st.title("📡 Disinfo Radar")
st.markdown(
    "**Setiap hoaks meninggalkan pola.** Aplikasi ini membaca judul ribuan artikel cek fakta Turnbackhoax, "
    "mengelompokkannya menjadi topik dengan TF-IDF + NMF, lalu menunjukkan topik mana yang naik, kata apa yang "
    "sedang ramai, dan hoaks lama mana yang paling mirip dengan klaim baru."
)

akhir = df["tanggal"].max()
bln_ini = df[df["tanggal"] > akhir - timedelta(days=30)]
bln_lalu = df[(df["tanggal"] <= akhir - timedelta(days=30)) & (df["tanggal"] > akhir - timedelta(days=60))]
porsi_penipuan = 100 * (df["label_kelompok"] == "Penipuan").mean()
porsi_penipuan_12 = 100 * (df[df["tanggal"] > akhir - timedelta(days=365)]["label_kelompok"] == "Penipuan").mean()
kat_top = df[df["kategori"] != "Lainnya"]["kategori"].value_counts()
if kat_top.empty:
    kat_top = df["kategori"].value_counts()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Artikel cek fakta", f"{len(df):,}", help="Jumlah artikel dalam filter yang dipilih.")
m2.metric("30 hari terakhir", f"{len(bln_ini):,}", delta=f"{len(bln_ini) - len(bln_lalu):+d} vs 30 hari sebelumnya",
          help=f"Dihitung dari artikel terakhir di arsip ({akhir:%d %b %Y}).")
m3.metric("Topik terbesar", URUTAN_TOPIK[0].split(" · ")[0], f"{ringkas.iloc[0]['persen']}% artikel · {URUTAN_TOPIK[0]}",
          delta_color="off", help=f"Klaster terbesar: {URUTAN_TOPIK[0]} ({ringkas.iloc[0]['jumlah']:,} artikel).")
m4.metric("Kategori teratas", kat_top.index[0], f"{100 * kat_top.iloc[0] / len(df):.0f}% artikel", delta_color="off",
          help="Kategori resmi Turnbackhoax; artikel tanpa kategori ('Lainnya') tidak dihitung di sini.")
m5.metric("Porsi penipuan (scam)", f"{porsi_penipuan_12:.0f}%", f"{porsi_penipuan_12 - porsi_penipuan:+.0f} poin vs rata-rata",
          help="Artikel berlabel [PENIPUAN] dalam 12 bulan terakhir dibanding rata-rata seluruh rentang.")

tab_radar, tab_topik, tab_kata, tab_cek, tab_modus, tab_data, tab_cara = st.tabs(
    ["📡 Radar", "🧩 Topic clusters", "📈 Keyword trends", "🔍 Similarity check", "🎭 Modus", "📋 Data & Sources", "🧭 How it works"]
)

# ---------------------------------------------------------------- 4a. RADAR
with tab_radar:
    c1, c2 = st.columns([3, 2])
    with c1:
        tren_label = analisis.tren_bulanan(df, "label_kelompok")
        urut = [l for l in URUTAN_LABEL if l in tren_label.columns]
        fig = px.bar(tren_label.reset_index().melt("bulan", var_name="Label", value_name="Artikel"),
                     x="bulan", y="Artikel", color="Label", color_discrete_map=WARNA_LABEL, category_orders={"Label": urut},
                     title="Artikel cek fakta per bulan, menurut label")
        fig.update_layout(bargap=0.15, xaxis_title=None)
        st.plotly_chart(rapikan(fig, 400), width="stretch")
    with c2:
        tren_topik = analisis.tren_bulanan(df, "topik")
        porsi = tren_topik.div(tren_topik.sum(axis=1).replace(0, 1), axis=0)
        # tiga topik dengan pertumbuhan porsi terbesar: rata-rata 6 bulan terakhir vs 6 bulan sebelumnya
        if len(porsi) >= 12:
            naik = (porsi.tail(6).mean() - porsi.iloc[-12:-6].mean()).sort_values(ascending=False)
        else:
            naik = porsi.mean().sort_values(ascending=False)
        naik = naik[naik > 0]
        tabel_naik = pd.DataFrame({"Topik": naik.index[:6], "Perubahan porsi": naik.values[:6] * 100})
        fig = px.bar(tabel_naik, x="Perubahan porsi", y="Topik", orientation="h", color="Topik", color_discrete_map=WARNA_TOPIK,
                     title="Topik yang porsinya paling naik (6 bulan terakhir vs 6 bulan sebelumnya)",
                     text=tabel_naik["Perubahan porsi"].map(lambda v: f"{v:+.1f}"))
        fig.update_layout(yaxis=dict(autorange="reversed", title=None, tickfont=dict(size=11)), xaxis_title="poin persen", showlegend=False,
                          xaxis=dict(range=[0, max(1.0, tabel_naik["Perubahan porsi"].max() * 1.25)]))
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(rapikan(fig, 400), width="stretch")

    st.subheader("Kata yang paling sering muncul")
    jendela = st.radio("Periode word cloud", ["12 bulan terakhir", "Seluruh rentang terpilih"], horizontal=True)
    df_wc = df[df["tanggal"] > akhir - timedelta(days=365)] if jendela.startswith("12") else df
    frek = analisis.frekuensi_kata(df_wc["teks_tfidf"], n=150)
    c1, c2 = st.columns([3, 2])
    with c1:
        if frek:
            wc = WordCloud(width=1100, height=520, background_color="#0b0f17", colormap="cool", max_words=150,
                           prefer_horizontal=0.9, random_state=7).generate_from_frequencies(frek)
            st.image(wc.to_array(), width="stretch",
                     caption=f"Word cloud {len(df_wc):,} judul ({jendela.lower()}); ukuran = jumlah judul yang memuat kata itu.")
    with c2:
        top = pd.DataFrame(list(frek.items())[:20], columns=["Kata", "Judul"])
        fig = px.bar(top, x="Judul", y="Kata", orientation="h", color_discrete_sequence=[WARNA_UTAMA], title="20 kata teratas")
        fig.update_layout(yaxis=dict(autorange="reversed", title=None), showlegend=False)
        st.plotly_chart(rapikan(fig, 520), width="stretch")

    st.subheader("🔴 Terbaru dari Turnbackhoax (RSS live)")
    kunci_waktu = datetime.utcnow().strftime("%Y-%m-%d %H:") + str(datetime.utcnow().minute // 30)
    rss = ambil_rss(kunci_waktu)
    if rss.sumber == "live" and not rss.data.empty:
        baru = rss.data.copy()
        baru["Topik terdekat"] = [model.nama[k] for k in analisis.prediksi_klaster(model, baru["judul"].tolist())]
        baru["Sudah di arsip"] = baru["id"].isin(df_arsip["id"]).map({True: "ya", False: "belum"})
        tampil = pd.DataFrame({
            "Tanggal": baru["tanggal"].dt.strftime("%d/%m/%Y"),
            "Judul": baru["judul"],
            "Topik terdekat": baru["Topik terdekat"],
            "Sudah di arsip": baru["Sudah di arsip"],
            "URL": baru["url"],
        })
        st.dataframe(tampil, hide_index=True, width="stretch", column_config=KONFIG_TABEL)
        st.caption("Setiap judul baru langsung dipetakan ke klaster terdekat oleh model yang sama. Sumber: turnbackhoax.id/feed, diperbarui tiap 30 menit.")
    else:
        st.info(f"RSS Turnbackhoax tidak terjangkau saat ini ({rss.pesan or 'tanpa pesan'}). Arsip tetap tersedia.")

    if mode_belajar:
        with st.expander("🎓 Dari judul mentah ke angka"):
            st.markdown(
                """
Setiap judul melewati `src/teks.py` sebelum dihitung:

```python
"[SALAH] Video Prabowo Umumkan Pertalite Naik Jadi Rp13.000"
 -> buang_label()  -> "Video Prabowo Umumkan Pertalite Naik Jadi Rp13.000"
 -> normalisasi()  -> "video prabowo umumkan pertalite naik jadi rp"
 -> tokenisasi()   -> ["prabowo", "umumkan", "pertalite"]      # stopword umum + domain dibuang
```

- **Stopword umum**: kata fungsi (yang, dan, akan, ...). **Stopword domain**: kata yang ada di hampir semua judul
  cek fakta (hoaks, video, foto, akun, beredar, faktanya) sehingga tidak membedakan topik.
- **Tanpa stemming**: pemotong imbuhan tanpa kamus merusak nama dan istilah (pemilu, menteri). Lihat Latihan 6.
- Grafik batang per bulan hanya `groupby(bulan, label).size()`; word cloud memakai jumlah **judul** yang memuat kata,
  bukan jumlah kemunculan, agar satu judul yang mengulang kata tidak mendominasi.
"""
            )

# ---------------------------------------------------------------- 4b. TOPIC CLUSTERS
with tab_topik:
    st.caption(f"{metode} mengelompokkan {len(df):,} judul menjadi {k_klaster} topik berdasarkan vektor TF-IDF. "
               "Nama topik = tiga kata berbobot terbesar di komponennya. Ganti metode atau jumlah topik di sidebar.")
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(ringkas, x="jumlah", y="nama", orientation="h", color="nama", color_discrete_map=WARNA_TOPIK,
                     title="Ukuran setiap topik", text="persen", hover_data={"kata_kunci": True, "kategori_dominan": True, "tahun_puncak": True})
        fig.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis_title="artikel", showlegend=False)
        st.plotly_chart(rapikan(fig, 30 * k_klaster + 150), width="stretch")
    with c2:
        tahunan = df.pivot_table(index="tahun", columns="topik", values="id", aggfunc="count", fill_value=0)
        porsi_thn = tahunan.div(tahunan.sum(axis=1), axis=0) * 100
        fig = px.area(porsi_thn.reset_index().melt("tahun", var_name="Topik", value_name="Porsi"),
                      x="tahun", y="Porsi", color="Topik", color_discrete_map=WARNA_TOPIK, category_orders={"Topik": URUTAN_TOPIK},
                      title="Porsi topik per tahun (%)")
        fig.update_layout(xaxis=dict(dtick=1, title=None), yaxis_title="% artikel", showlegend=False)
        st.plotly_chart(rapikan(fig, 30 * k_klaster + 150), width="stretch")
        st.caption("Warna mengikuti grafik di kiri. Arahkan kursor untuk nama topik.")

    st.dataframe(
        ringkas[["nama", "jumlah", "persen", "kategori_dominan", "tahun_puncak", "label_salah_persen", "kata_kunci"]].rename(columns={
            "nama": "Topik", "jumlah": "Artikel", "persen": "%", "kategori_dominan": "Kategori dominan",
            "tahun_puncak": "Tahun puncak", "label_salah_persen": "% label Salah", "kata_kunci": "Kata kunci (bobot tertinggi)"}),
        hide_index=True, width="stretch",
        column_config={"%": st.column_config.NumberColumn(format="%.1f"), "% label Salah": st.column_config.NumberColumn(format="%.0f")},
    )

    st.subheader("Telusuri satu topik")
    topik_pilih = st.selectbox("Topik", URUTAN_TOPIK)
    id_klaster = int(ringkas.loc[ringkas["nama"] == topik_pilih, "klaster"].iloc[0])
    sub = df[df["klaster"] == id_klaster]
    c1, c2 = st.columns([2, 3])
    with c1:
        kk = pd.DataFrame(model.kata_kunci[id_klaster], columns=["Kata", "Bobot"])
        fig = px.bar(kk, x="Bobot", y="Kata", orientation="h", color_discrete_sequence=[WARNA_TOPIK[topik_pilih]],
                     title="Kata kunci topik (bobot komponen)")
        fig.update_layout(yaxis=dict(autorange="reversed", title=None), showlegend=False)
        st.plotly_chart(rapikan(fig, 380), width="stretch")
        per_bulan = sub.groupby("bulan").size().rename("Artikel").reset_index()
        fig = px.line(per_bulan, x="bulan", y="Artikel", title=f"Artikel per bulan: {topik_pilih}", color_discrete_sequence=[WARNA_TOPIK[topik_pilih]])
        fig.update_layout(xaxis_title=None)
        st.plotly_chart(rapikan(fig, 300), width="stretch")
    with c2:
        st.markdown(f"**{len(sub):,} artikel**, {sub['tanggal'].min():%b %Y} – {sub['tanggal'].max():%b %Y}. Contoh terbaru:")
        st.dataframe(tautan_judul(sub.head(40)), hide_index=True, width="stretch", height=640, column_config=KONFIG_TABEL)

    if mode_belajar:
        with st.expander("🎓 TF-IDF, NMF, dan K-Means dalam lima menit"):
            st.markdown(
                r"""
**TF-IDF** memberi bobot setiap kata di setiap judul:

$$\text{tfidf}(k, d) = \underbrace{(1 + \log \text{tf}_{k,d})}_{\text{sering di judul ini}} \times \underbrace{\log\frac{1 + N}{1 + \text{df}_k} + 1}_{\text{jarang di judul lain}}$$

`tf` = berapa kali kata *k* muncul di judul *d*; `df` = berapa judul yang memuat kata *k*; *N* = jumlah judul.
Kata seperti "prabowo" mendapat bobot besar di judul yang memuatnya karena jarang di judul lain; kata yang muncul
di separuh judul dibuang (`max_df=0.5`). Setiap vektor judul dinormalisasi ke panjang 1.

```python
vec = TfidfVectorizer(tokenizer=str.split, ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True)
X = vec.fit_transform(judul_bersih)          # matriks jarang: judul x (kata + pasangan kata)
```

**NMF** (Non-negative Matrix Factorization, metode bawaan) memecah matriks TF-IDF menjadi dua matriks kecil
yang semuanya non-negatif:

$$X_{(\text{judul} \times \text{kata})} \approx W_{(\text{judul} \times \text{topik})} \cdot H_{(\text{topik} \times \text{kata})}$$

Baris *H* adalah topik: kata dengan bobot terbesar di baris itu menjadi nama topik. Baris *W* adalah "resep"
setiap judul: seberapa banyak ia mengandung tiap topik. Aplikasi mengambil topik dengan bobot terbesar
(`W.argmax(axis=1)`) sebagai label judul.

```python
nmf = NMF(n_components=k, init="nndsvda", random_state=42).fit(X)
W = nmf.transform(X)              # judul x topik
label = W.argmax(axis=1)          # topik dominan per judul
kata_kunci = nmf.components_      # H: topik x kata
```

**K-Means** (pilihan kedua di sidebar) mencari *k* titik pusat yang meminimalkan jarak kuadrat setiap judul ke
pusat terdekatnya: pilih *k* pusat awal (k-means++), tempelkan setiap judul ke pusat terdekat, geser pusat ke
rata-rata anggotanya, ulangi sampai tidak berubah. Karena vektor berpanjang 1, jarak Euclid setara dengan
*cosine similarity*; nama klaster diambil dari koordinat pusat terbesar.

Mengapa NMF menjadi bawaan? Coba ganti ke K-Means di sidebar dan lihat ukuran klasternya: pada judul pendek,
K-Means biasanya menumpuk 60-70% judul di satu klaster "keranjang sisa" karena memaksa setiap judul masuk tepat
satu klaster berdasarkan jarak. NMF memberi bobot campuran, sehingga judul yang menyinggung dua topik tetap
terwakili dan ukuran topik jauh lebih seimbang. Latihan 7 memperbaiki K-Means dengan LSA (TruncatedSVD).
"""
            )

# ---------------------------------------------------------------- 4c. KEYWORD TRENDS
with tab_kata:
    st.caption("Berapa judul per bulan yang memuat kata tertentu. Kata dicocokkan utuh pada teks yang sudah dibersihkan.")
    frek_semua = analisis.frekuensi_kata(df["teks_tfidf"], n=400)
    bawaan = [k for k in ["vaksin", "gempa", "prabowo", "jokowi", "bansos", "pemilu", "china", "israel"] if k in frek_semua][:6]
    c1, c2 = st.columns([3, 1])
    kata_pilih = c1.multiselect("Kata yang diikuti", list(frek_semua.keys()), default=bawaan)
    kata_bebas = c2.text_input("Tambah kata lain", placeholder="misal: megathrust")
    if kata_bebas.strip():
        kata_pilih = kata_pilih + [k.strip() for k in kata_bebas.split(",") if k.strip()]
    halus = st.toggle("Haluskan (rata-rata bergerak 3 bulan)", value=True)
    if kata_pilih:
        tk = analisis.tren_kata(df, kata_pilih)
        if halus and len(tk) >= 3:
            tk = tk.rolling(3, min_periods=1).mean()
        fig = px.line(tk.reset_index().melt("bulan", var_name="Kata", value_name="Judul per bulan"),
                      x="bulan", y="Judul per bulan", color="Kata", color_discrete_sequence=PALET, title="Tren kata per bulan")
        fig.update_layout(xaxis_title=None)
        st.plotly_chart(rapikan(fig, 440), width="stretch")
    else:
        st.info("Pilih minimal satu kata.")

    st.subheader("Kata yang naik dan turun")
    bulan_batas = st.slider("Bandingkan N bulan terakhir dengan periode sebelumnya", 3, 36, 12)
    batas = (akhir - pd.DateOffset(months=bulan_batas)).normalize()
    nt = analisis.kata_naik_turun(df, batas, min_total=8, n=15)
    if nt.empty:
        st.info("Rentang terlalu pendek untuk dibandingkan.")
    else:
        c1, c2 = st.columns(2)
        for kolom, arah, warna, judul in [(c1, "Naik", "#4ade80", f"📈 Makin ramai sejak {batas:%b %Y}"),
                                          (c2, "Turun", "#ff5c8a", f"📉 Mereda sejak {batas:%b %Y}")]:
            sub_nt = nt[nt["arah"] == arah].copy()
            sub_nt["label"] = sub_nt.apply(lambda r: f"{r['per_1000_sebelum']:.0f} → {r['per_1000_sesudah']:.0f} per 1.000 judul", axis=1)
            fig = px.bar(sub_nt, x="rasio", y="kata", orientation="h", color_discrete_sequence=[warna], title=judul,
                         text="label", hover_data={"sebelum": True, "sesudah": True})
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis_title="rasio laju sesudah / sebelum",
                              showlegend=False, xaxis_type="log")
            kolom.plotly_chart(rapikan(fig, 520), width="stretch")

    if mode_belajar:
        with st.expander("🎓 Cara mengukur kata yang 'naik'"):
            st.markdown(
                r"""
Jumlah mentah menyesatkan karena jumlah artikel per bulan berubah (2020 jauh lebih ramai dari 2017). Yang
dibandingkan adalah **laju per 1.000 judul** sebelum dan sesudah tanggal batas, dengan pemulusan Laplace (+1)
supaya kata yang baru muncul tidak dibagi nol:

$$\text{rasio} = \frac{(f_{\text{sesudah}} + 1)/(N_{\text{sesudah}} + 1)}{(f_{\text{sebelum}} + 1)/(N_{\text{sebelum}} + 1)}$$

Hanya kata dengan total ≥ 8 kemunculan yang ikut, agar nama langka yang muncul tiga kali tidak tampil sebagai
"naik 300%". Sumbu x memakai skala log supaya kenaikan 10x dan penurunan 10x terlihat simetris.
`analisis.kata_naik_turun` berisi seluruh perhitungannya (sekitar 25 baris pandas + Counter).
"""
            )

# ---------------------------------------------------------------- 4d. SIMILARITY CHECK
with tab_cek:
    st.markdown(
        "Tempel sebuah klaim atau judul berita yang mencurigakan. Aplikasi mencari artikel cek fakta yang **paling mirip** "
        "di arsip dan menunjukkan topik terdekatnya. **Ini bukan vonis benar/salah**: hoaks sering lahir kembali dengan kata "
        "yang sama, dan daftar ini membantu menemukan pemeriksaan yang sudah pernah dilakukan."
    )
    contoh = ["Pemerintah akan menghapus BPJS Kesehatan dan menggantinya dengan klinik koperasi desa",
              "BMKG memprediksi gempa megathrust magnitudo 9 akan terjadi minggu depan",
              "Vaksin baru mengandung chip yang bisa melacak lokasi penerimanya",
              "Pendaftaran bantuan sosial Rp2 juta lewat tautan WhatsApp"]
    klaim = st.text_area("Klaim yang ingin dicek", value=contoh[0], height=90)
    c1, c2 = st.columns([1, 3])
    n_mirip = c1.slider("Jumlah hasil", 5, 30, 10)
    if c2.button("Gunakan contoh acak"):
        import random
        klaim = random.choice(contoh)
        st.rerun()
    if klaim.strip():
        idx, skor = analisis.cari_mirip(model, klaim, n=n_mirip)
        k_pred = int(analisis.prediksi_klaster(model, [klaim])[0])
        token = teks.tokenisasi(teks.buang_label(klaim))
        dikenal = [t for t in token if t in model.vectorizer.vocabulary_]
        st.markdown(f"**Topik terdekat:** `{model.nama[k_pred]}` &nbsp;·&nbsp; kata yang dikenali model: "
                    + (", ".join(f"`{t}`" for t in dikenal) if dikenal else "*(tidak ada; hasil akan acak)*"))
        hasil = df.loc[idx].copy()
        hasil["Kemiripan"] = skor
        hasil = hasil[hasil["Kemiripan"] > 0]
        if hasil.empty:
            st.warning("Tidak ada judul di arsip yang berbagi kata dengan klaim ini. Coba kata yang lebih umum.")
        else:
            tampil = tautan_judul(hasil)
            tampil.insert(0, "Kemiripan", hasil["Kemiripan"].values)
            tampil.insert(4, "Topik", hasil["topik"].values)
            st.dataframe(tampil, hide_index=True, width="stretch",
                         column_config={**KONFIG_TABEL, "Kemiripan": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
            tertinggi = hasil.iloc[0]
            st.caption(f"Paling mirip: \"{tertinggi['judul_bersih']}\" ({tertinggi['tanggal']:%d %b %Y}, label {tertinggi['label_kelompok']}), "
                       f"cosine similarity {tertinggi['Kemiripan']:.2f}. Nilai 1 berarti kata-katanya identik.")

    if mode_belajar:
        with st.expander("🎓 Cosine similarity: kenapa cukup perkalian titik"):
            st.markdown(
                r"""
Klaim baru diubah ke vektor TF-IDF dengan kosakata model yang **sama** (`vectorizer.transform`), lalu dibandingkan
dengan semua judul arsip:

$$\cos(\theta) = \frac{\vec a \cdot \vec b}{\lVert \vec a \rVert \, \lVert \vec b \rVert}$$

Karena TfidfVectorizer menormalkan setiap vektor ke panjang 1, penyebutnya 1 dan cosine = perkalian titik.
Satu operasi matriks jarang `X @ v.T` menghasilkan skor untuk seluruh arsip sekaligus:

```python
v = model.vectorizer.transform([teks.bersihkan_judul(klaim)])
skor = (model.X @ v.T).toarray().ravel()      # satu angka per judul arsip
urut = np.argsort(skor)[::-1][:n]
```

Kata yang tidak ada di kosakata (nama baru, salah ketik) diabaikan; itulah sebabnya daftar "kata yang dikenali
model" ditampilkan. Pendekatan ini hanya mengukur kesamaan **kata**, bukan makna: "harga BBM naik" dan "BBM makin
mahal" dianggap berbeda. Latihan 8 mengganti TF-IDF dengan *sentence embedding* untuk menangkap sinonim.
"""
            )

# ---------------------------------------------------------------- 4e. MODUS
with tab_modus:
    st.caption("Bagaimana bentuk hoaks berubah: label pemeriksa fakta, format unggahan yang disebut di judul, dan kategori per tahun.")
    c1, c2 = st.columns(2)
    with c1:
        thn_label = df.pivot_table(index="tahun", columns="label_kelompok", values="id", aggfunc="count", fill_value=0)
        thn_label = thn_label.div(thn_label.sum(axis=1), axis=0) * 100
        urut = [l for l in URUTAN_LABEL if l in thn_label.columns]
        fig = px.bar(thn_label[urut].reset_index().melt("tahun", var_name="Label", value_name="Porsi"),
                     x="tahun", y="Porsi", color="Label", color_discrete_map=WARNA_LABEL, category_orders={"Label": urut},
                     title="Porsi label per tahun (%)")
        fig.update_layout(xaxis=dict(dtick=1, title=None), yaxis_title="% artikel", bargap=0.2)
        st.plotly_chart(rapikan(fig, 400), width="stretch")
    with c2:
        thn_fmt = df.pivot_table(index="tahun", columns="format", values="id", aggfunc="count", fill_value=0)
        thn_fmt = thn_fmt.div(thn_fmt.sum(axis=1), axis=0) * 100
        urut_f = [f for f in WARNA_FORMAT if f in thn_fmt.columns]
        fig = px.bar(thn_fmt[urut_f].reset_index().melt("tahun", var_name="Format", value_name="Porsi"),
                     x="tahun", y="Porsi", color="Format", color_discrete_map=WARNA_FORMAT, category_orders={"Format": urut_f},
                     title="Format unggahan yang disebut di judul, per tahun (%)")
        fig.update_layout(xaxis=dict(dtick=1, title=None), yaxis_title="% artikel", bargap=0.2)
        st.plotly_chart(rapikan(fig, 400), width="stretch")

    thn_kat = df.pivot_table(index="kategori", columns="tahun", values="id", aggfunc="count", fill_value=0)
    thn_kat = thn_kat.div(thn_kat.sum(axis=0), axis=1) * 100
    thn_kat = thn_kat.loc[thn_kat.sum(axis=1).sort_values(ascending=False).index]
    fig = px.imshow(thn_kat, color_continuous_scale="Purples", aspect="auto", text_auto=".0f",
                    title="Kategori per tahun (% dari artikel tahun itu)", labels=dict(color="%"))
    fig.update_layout(xaxis=dict(dtick=1, title=None, side="top"), yaxis_title=None, coloraxis_showscale=False)
    st.plotly_chart(rapikan(fig, 40 * len(thn_kat) + 120), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        hari = df["tanggal"].dt.dayofweek.value_counts().sort_index()
        nama_hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
        fig = px.bar(x=[nama_hari[i] for i in hari.index], y=hari.values, color_discrete_sequence=[WARNA_UTAMA],
                     title="Hari terbit artikel cek fakta", labels={"x": "", "y": "artikel"})
        st.plotly_chart(rapikan(fig, 340), width="stretch")
    with c2:
        kat_label = df.pivot_table(index="kategori", columns="label_kelompok", values="id", aggfunc="count", fill_value=0)
        kat_label = kat_label.loc[kat_label.sum(axis=1).sort_values(ascending=False).index[:12]]
        urut = [l for l in URUTAN_LABEL if l in kat_label.columns]
        fig = px.bar(kat_label[urut].reset_index().melt("kategori", var_name="Label", value_name="Artikel"),
                     x="Artikel", y="kategori", color="Label", orientation="h", color_discrete_map=WARNA_LABEL,
                     category_orders={"Label": urut}, title="Label per kategori (12 kategori terbesar)")
        fig.update_layout(yaxis=dict(autorange="reversed", title=None))
        st.plotly_chart(rapikan(fig, 340), width="stretch")

    if mode_belajar:
        with st.expander("🎓 Dari mana label dan format berasal"):
            st.markdown(
                """
- **Label** diambil dari awalan judul: `[SALAH]`, `[PENIPUAN]`, `[KLARIFIKASI]`, `[BENAR]`, dan gaya lama seperti
  `(HOAX)` atau `[DISINFORMASI]`. `muat_data.kelompokkan_label` menyatukan gaya-gaya itu ke enam kelompok.
  Mafindo mulai memakai label *Penipuan* untuk tautan phishing, lowongan palsu, dan undian bodong; kenaikan
  porsinya adalah salah satu temuan paling jelas di arsip ini.
- **Format** ditebak dari kata di judul (`teks.deteksi_format`): "video", "foto", "tautan", "pesan berantai".
  Ini tebakan kasar: judul yang tidak menyebut medium masuk "Teks / lainnya". Cukup untuk melihat arah,
  tidak cukup untuk angka pasti.
- **Kategori** adalah label resmi Turnbackhoax (Politik, Kesehatan, Bencana, ...). Artikel lama banyak yang
  tanpa kategori, sehingga "Lainnya" besar di tahun-tahun awal.
"""
            )

# ---------------------------------------------------------------- 4f. DATA & SOURCES
with tab_data:
    st.markdown(
        """
| Data | Sumber | Keterangan |
|---|---|---|
| Arsip artikel cek fakta | [turnbackhoax.id](https://turnbackhoax.id) (Mafindo), halaman daftar artikel publik | `data/hoaks_turnbackhoax.csv`: id, tanggal, judul, label, kategori, ringkasan singkat, URL. Dibangun dengan `scripts/kumpulkan_data.py` |
| Artikel terbaru | [turnbackhoax.id/feed](https://turnbackhoax.id/feed) (RSS resmi) | Diambil saat aplikasi dibuka, cache 30 menit |
| Stopword | `src/teks.py` | Daftar stopword Indonesia umum + stopword domain cek fakta, disusun sendiri |

Yang **tidak** ada di sini: isi lengkap artikel, gambar, dan penjelasan pemeriksa fakta. Untuk itu buka tautan artikelnya.
Rincian lisensi dan keterbatasan ada di [`data/SUMBER_DATA.md`](https://github.com/Avalon2222/disinfo-radar/blob/main/data/SUMBER_DATA.md).
"""
    )
    cari = st.text_input("Cari di judul", placeholder="misal: pertalite")
    tampil = df if not cari.strip() else df[df["judul_bersih"].str.contains(cari.strip(), case=False, regex=False)]
    st.markdown(f"**{len(tampil):,} artikel** dalam filter saat ini" + (f" yang memuat \"{cari.strip()}\"" if cari.strip() else ""))
    tabel = tautan_judul(tampil)
    tabel.insert(4, "Topik", tampil["topik"].values)
    tabel.insert(5, "Format", tampil["format"].values)
    st.dataframe(tabel, hide_index=True, width="stretch", height=480, column_config=KONFIG_TABEL)
    unduh = tampil[["id", "tanggal", "judul", "label", "label_kelompok", "kategori", "topik", "format", "ringkasan", "url"]]
    st.download_button("⬇️ Download CSV (filter saat ini + topik)", unduh.to_csv(index=False).encode("utf-8"),
                       "disinfo_radar.csv", "text/csv", width="stretch")
    st.caption(DISCLAIMER)

# ---------------------------------------------------------------- 4g. HOW IT WORKS
with tab_cara:
    st.subheader("Alur data → teks → topik → grafik")
    st.graphviz_chart(
        """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#161c28", fontcolor="#e8e8e8", color="#55607a", fontname="sans-serif", fontsize=11];
  edge [color="#8899aa", fontcolor="#aab4c4"];
  tbh  [label="turnbackhoax.id\\nhalaman daftar artikel"];
  rss  [label="turnbackhoax.id/feed\\nRSS 10 artikel terakhir"];
  kump [label="scripts/kumpulkan_data.py\\nparse kartu -> CSV"];
  csv  [label="data/hoaks_turnbackhoax.csv\\nid · tanggal · judul · label · kategori"];
  muat [label="src/muat_data.py\\nlabel -> kelompok, kolom turunan"];
  tks  [label="src/teks.py\\nbuang label, stopword, tokenisasi"];
  ana  [label="src/analisis.py\\nTF-IDF -> NMF / K-Means -> nama topik\\ntren · kata naik · cosine similarity"];
  app  [label="app.py\\nsidebar, tab, grafik Plotly, word cloud"];
  user [label="Browser\\n(Streamlit)", fillcolor="#2a2140"];
  tbh -> kump -> csv -> muat -> tks -> ana -> app -> user; rss -> muat; user -> app [label="filter -> rerun", fontsize=9];
}
"""
    )
    st.markdown(
        """
| Langkah | File | Yang terjadi |
|---|---|---|
| 1 | `scripts/kumpulkan_data.py` | Membaca halaman daftar artikel Turnbackhoax (10 kartu per halaman), mengambil id, judul, ringkasan, kategori, tanggal. Mode perbarui hanya mengambil yang baru. |
| 2 | `src/muat_data.py` | Membaca CSV, menyatukan gaya label (`[SALAH]`, `(HOAX)`, ...) ke enam kelompok, menambah kolom bulan/tahun/format. Mengambil RSS untuk artikel terbaru. |
| 3 | `src/teks.py` | Membersihkan judul: buang awalan label, URL, angka, tanda baca; buang stopword umum dan stopword domain. |
| 4 | `src/analisis.py` | **TF-IDF** (kata + pasangan kata) → **NMF** (atau K-Means) k topik → nama topik dari komponen → tren bulanan, kata naik/turun, cosine similarity untuk klaim baru. |
| 5 | `app.py` | Filter sidebar → model di-cache per (filter, k, metode) → tujuh tab. |

**Konsep yang dipakai**

- **Bag of words & TF-IDF**: teks menjadi vektor; bobot besar untuk kata yang sering di satu dokumen tetapi jarang di dokumen lain.
- **N-gram**: pasangan kata ("kartu prakerja") sebagai fitur tambahan.
- **NMF**: faktorisasi matriks non-negatif, X ≈ W·H; baris H = topik, argmax W = topik dominan tiap judul.
- **K-Means**: klasterisasi tak terawasi sebagai pembanding; nama klaster dari koordinat pusat.
- **Cosine similarity**: kemiripan arah vektor, cocok untuk teks pendek.
- **Laju per 1.000 + pemulusan Laplace**: membandingkan frekuensi kata antar periode yang ukurannya berbeda.
- **Caching Streamlit**: `cache_data` untuk tabel, `cache_resource` untuk model sklearn.
"""
    )
    st.caption(DISCLAIMER)
