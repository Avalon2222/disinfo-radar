# 🏋️ Latihan

Diurutkan dari yang mudah ke yang lebih menantang. Setiap latihan dilengkapi petunjuk dan kunci jawaban yang dapat
dibuka. Usahakan mencoba minimal 10 menit sebelum membuka kunci. Setelah setiap latihan, jalankan
`python -m pytest tests/` dan `streamlit run app.py`.

---

### Latihan 1 — Ganti warna label ⭐

Warna label "Penipuan" saat ini amber (`#ffb000`). Ubah menjadi oranye `#fb923c` dan pastikan grafik batang bulanan,
grafik porsi label per tahun, dan grafik label per kategori ikut berubah tanpa menyentuh `app.py`.

<details><summary>Petunjuk</summary>
Semua warna label didefinisikan di satu tempat: `WARNA_LABEL` di `src/muat_data.py`. `app.py` hanya mengimpornya dan
memakainya lewat `color_discrete_map`.
</details>

<details><summary>Kunci</summary>

```python
WARNA_LABEL = {
    "Salah": "#ff5c8a", "Penipuan": "#fb923c", ...
}
```
Karena semua grafik label memakai `color_discrete_map=WARNA_LABEL`, perubahan satu baris berlaku di mana-mana.
</details>

---

### Latihan 2 — Tambah stopword domain ⭐

Buka tab Radar dan perhatikan 20 kata teratas. Ada kata yang menurutmu tidak menceritakan topik (misalnya nama
platform atau kata kerja umum yang lolos)? Tambahkan ke `STOPWORD_DOMAIN` di `src/teks.py`, lalu bandingkan
word cloud sebelum dan sesudah.

<details><summary>Petunjuk</summary>
`STOPWORD_DOMAIN` adalah `set` dari string yang dipisah spasi. Tambahkan kata dalam huruf kecil. Tes
`test_tokenisasi_membuang_stopword_umum_dan_domain` memastikan kata lama tetap terbuang.
</details>

<details><summary>Kunci</summary>
Tidak ada jawaban tunggal. Pertanyaan yang membantu: kalau kata ini muncul di dua judul, apakah keduanya kemungkinan
besar membahas hal yang sama? Kalau tidak, ia stopword. Ingat: kata yang dibuang di sini juga hilang dari pencarian
kemiripan dan tren kata.
</details>

---

### Latihan 3 — Filter tanggal yang lebih halus ⭐⭐

Slider sidebar hanya memfilter per tahun. Ganti dengan pemilih rentang tanggal (`st.date_input` dengan dua tanggal)
sehingga pengguna bisa melihat, misalnya, hanya masa kampanye Pilpres 2024 (November 2023–Februari 2024).

<details><summary>Petunjuk</summary>
`st.date_input("Rentang", value=(awal, akhir), min_value=..., max_value=...)` mengembalikan tuple dua `date`.
Filter `df_arsip` dengan `df_arsip["tanggal"].dt.date.between(awal, akhir)`. Perhatikan: `bangun_model` di-cache
dengan kunci `ids`, sehingga filter baru otomatis memicu model baru.
</details>

<details><summary>Kunci</summary>

```python
awal, akhir = st.date_input("Rentang tanggal", value=(df_arsip["tanggal"].min().date(), df_arsip["tanggal"].max().date()))
df = df_arsip[df_arsip["tanggal"].dt.date.between(awal, akhir) & df_arsip["kategori"].isin(kategori_pilih)]
```
Tangani kasus pengguna baru memilih satu tanggal (tuple berisi satu elemen) dengan `if len(rentang) != 2: st.stop()`.
</details>

---

### Latihan 4 — Topik per kategori ⭐⭐

Tambahkan tabel atau heatmap di tab Topic clusters yang menunjukkan **kategori × topik**: berapa artikel kategori
"Politik" masuk topik mana saja. Apakah ada topik yang menyebar ke banyak kategori? Itu kandidat "keranjang sisa".

<details><summary>Petunjuk</summary>
`df.pivot_table(index="kategori", columns="topik", values="id", aggfunc="count", fill_value=0)` lalu `px.imshow`
seperti heatmap kategori per tahun di tab Modus.
</details>

<details><summary>Kunci</summary>

```python
kt = df.pivot_table(index="kategori", columns="topik", values="id", aggfunc="count", fill_value=0)
kt = kt.div(kt.sum(axis=1), axis=0) * 100          # porsi per kategori
fig = px.imshow(kt, color_continuous_scale="Purples", aspect="auto", text_auto=".0f")
```
Topik yang persennya merata di semua baris adalah topik yang kata kuncinya terlalu umum.
</details>

---

### Latihan 5 — Berapa k yang pas? Silhouette score ⭐⭐

Hitung *silhouette score* untuk k = 6, 8, 10, ..., 20 pada data terpilih dan tampilkan sebagai grafik garis di
learning mode tab Topic clusters. Silhouette (−1 sampai 1) mengukur seberapa dekat judul ke klasternya sendiri
dibanding klaster tetangga.

<details><summary>Petunjuk</summary>
`from sklearn.metrics import silhouette_score`; `silhouette_score(X, km.labels_, sample_size=3000, random_state=0)`.
Menghitung untuk semua pasangan judul mahal, karena itu `sample_size`. Bungkus dengan `@st.cache_data` dan jalankan
hanya bila tombol ditekan.
</details>

<details><summary>Kunci</summary>

```python
@st.cache_data(show_spinner="Menghitung silhouette...")
def kurva_silhouette(dokumen: tuple[str, ...], daftar_k: tuple[int, ...]) -> pd.DataFrame:
    vec, X = analisis.bangun_tfidf(list(dokumen))
    baris = []
    for k in daftar_k:
        km = analisis.klasterkan(X, k)
        baris.append({"k": k, "silhouette": silhouette_score(X, km.labels_, sample_size=3000, random_state=0)})
    return pd.DataFrame(baris)
```
Pada teks pendek nilainya kecil (0,02–0,08) dan sering tidak punya puncak tegas. Itu pelajaran penting: metrik
memberi petunjuk, bukan jawaban; kamu tetap harus membaca klasternya.
</details>

---

### Latihan 6 — Stemming dengan Sastrawi ⭐⭐⭐

Pasang `PySastrawi` dan tambahkan opsi stemming pada `teks.tokenisasi`. Latih ulang model dengan dan tanpa
stemming, lalu bandingkan nama klasternya. Apakah "vaksinasi" dan "vaksin" menyatu? Apakah ada kata yang rusak?

<details><summary>Petunjuk</summary>
```python
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
stemmer = StemmerFactory().create_stemmer()
stemmer.stem("vaksinasi")   # -> 'vaksin'
```
Sastrawi lambat (±1 ms per kata). Untuk belasan ribu judul, simpan hasil stem di cache `dict` atau pra-hitung sekali
dan simpan ke kolom baru di CSV.
</details>

<details><summary>Kunci</summary>
Tambahkan parameter `stem: bool = False` ke `tokenisasi`, dan terapkan stemmer **setelah** stopword dibuang (stopword
disusun dalam bentuk belum di-stem). Hasil yang umum terlihat: beberapa topik lebih padat (vaksin/vaksinasi/divaksin
menyatu), tetapi nama orang dan singkatan kadang berubah bentuk. Keputusan memakai stemming atau tidak bergantung
tujuan: untuk pencarian kemiripan biasanya membantu, untuk nama klaster yang mudah dibaca kadang mengganggu.
</details>

---

### Latihan 7 — Perbaiki K-Means dengan LSA ⭐⭐⭐

Ganti metode ke K-Means di sidebar dan lihat tab Topic clusters: satu klaster menampung 60-70% judul. Salah satu
obat klasik adalah **LSA**: kurangi dimensi TF-IDF menjadi ±100 komponen dengan `TruncatedSVD`, normalkan lagi,
baru jalankan K-Means. Tambahkan metode ketiga "K-Means + LSA" dan bandingkan ukuran klasternya.

<details><summary>Petunjuk</summary>
```python
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
svd = TruncatedSVD(100, random_state=42)
Z = Normalizer().fit_transform(svd.fit_transform(X))      # judul x 100, panjang 1
km = KMeans(k, n_init=5, random_state=42).fit(Z)
komponen = svd.inverse_transform(km.cluster_centers_)   # kembali ke ruang kata untuk kata kunci
```
`rakit_model` di `src/analisis.py` adalah tempat menambah cabang baru; `ModelTopik.prediksi` perlu tahu bahwa
vektor klaim baru harus melewati `svd.transform` dan `Normalizer` yang sama sebelum `km.predict`.
</details>

<details><summary>Kunci</summary>
Pada arsip ini, LSA 100 dimensi menurunkan klaster terbesar dari ±70% menjadi ±45% dan memunculkan topik yang
sebelumnya tertelan (undian berhadiah, wali kota/bupati, kesehatan). Masih kalah seimbang dari NMF, tetapi
menunjukkan mengapa reduksi dimensi membantu K-Means: dalam ruang 8.000 kata hampir semua judul "jauh" dari
semua pusat; dalam 100 dimensi padat, kedekatan menjadi bermakna. Untuk kata kunci, `svd.inverse_transform`
memetakan pusat kembali ke ruang kata; nilai negatif kecil bisa diabaikan.
</details>

---

### Latihan 8 — Kemiripan makna dengan sentence embedding ⭐⭐⭐⭐

TF-IDF menganggap "harga BBM naik" dan "BBM makin mahal" tidak mirip. Pasang `sentence-transformers`, pakai model
multibahasa kecil (`paraphrase-multilingual-MiniLM-L12-v2`), hitung embedding semua judul sekali (simpan ke `.npy`),
dan buat opsi "Metode kemiripan: TF-IDF / Embedding" di tab Similarity check.

<details><summary>Petunjuk</summary>
```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
E = m.encode(df["judul_bersih"].tolist(), normalize_embeddings=True, batch_size=128)   # (n, 384)
np.save("data/embedding_judul.npy", E)
skor = E @ m.encode([klaim], normalize_embeddings=True)[0]
```
Model ±470 MB; jangan dimasukkan ke repo. Di Streamlit Cloud, unduhan pertama butuh waktu dan memori; pertimbangkan
hanya mengaktifkannya secara lokal.
</details>

<details><summary>Kunci</summary>
Uji dengan pasangan yang sinonim ("meninggal" vs "tewas", "dibatalkan" vs "ditunda") dan pasangan yang berbagi kata
tetapi berbeda makna. Embedding menang di kasus pertama, TF-IDF sering lebih tajam di kasus nama spesifik. Gabungan
sederhana (rata-rata dua skor yang sudah dinormalkan) biasanya terbaik. Catat juga waktu encode: ini contoh nyata
*trade-off* kualitas vs biaya komputasi.
</details>

---

### Latihan 9 — Deteksi lonjakan otomatis ⭐⭐⭐⭐

Buat fungsi `deteksi_lonjakan(df, topik)` yang menandai bulan ketika jumlah artikel satu topik melampaui
rata-rata 12 bulan sebelumnya + 2 simpangan baku (z-score bergulir, seperti pendeteksi hari liar di proyek
btc-news-volatility). Tampilkan tanda di grafik tren topik dan daftar "lonjakan" dengan tiga judul contoh dari
bulan itu. Apakah lonjakan bertepatan dengan peristiwa nyata (pemilu, bencana, pandemi)?

<details><summary>Petunjuk</summary>
```python
s = df[df["topik"] == topik].groupby("bulan").size().reindex(semua_bulan, fill_value=0)
rata = s.rolling(12).mean().shift(1)
sd = s.rolling(12).std().shift(1)
z = (s - rata) / sd.replace(0, np.nan)
lonjakan = s[z > 2]
```
`shift(1)` penting: bulan ini tidak boleh ikut menghitung rata-rata pembandingnya sendiri.
</details>

<details><summary>Kunci</summary>
Tambahkan `fig.add_trace(go.Scatter(x=lonjakan.index, y=lonjakan.values, mode="markers", marker=dict(size=12, symbol="star")))`
pada grafik per bulan di bagian "Telusuri satu topik". Untuk judul contoh, ambil `df[(df["topik"] == topik) & (df["bulan"] == b)].head(3)`.
Perhatikan bahwa jumlah total artikel juga berfluktuasi; versi yang lebih adil memakai **porsi** topik per bulan, bukan jumlah.
Tulis tes dengan deret buatan: 12 bulan bernilai 5, lalu satu bulan bernilai 30.
</details>
