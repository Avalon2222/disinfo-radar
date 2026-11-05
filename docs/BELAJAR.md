# 📚 Belajar dari proyek ini

Panduan ini menjelaskan apa yang terjadi di balik aplikasi, dari satu judul artikel cek fakta sampai nama topik
yang muncul di grafik. Tidak perlu latar belakang NLP; cukup Python dasar dan rasa penasaran. Setiap bagian
menunjuk ke file dan fungsi yang bisa langsung dibaca.

---

## 1. Gambaran besar

```
turnbackhoax.id ─► scripts/kumpulkan_data.py ─► data/hoaks_turnbackhoax.csv
                                                       │
                                                       ▼
                                             src/muat_data.py  (label → kelompok, bulan, tahun, format)
                                                       │
                                                       ▼
                                             src/teks.py       (buang label, stopword, tokenisasi)
                                                       │
                                                       ▼
                                             src/analisis.py   (TF-IDF → NMF / K-Means → nama topik → tren → kemiripan)
                                                       │
                                                       ▼
                                             app.py            (sidebar, 7 tab, grafik Plotly, word cloud)
```

Empat pertanyaan yang dijawab aplikasi, dan di mana jawabannya dihitung:

| Pertanyaan | Fungsi |
|---|---|
| Hoaks di arsip ini **tentang apa saja**? | `analisis.bangun_model` + `analisis.ringkasan_klaster` |
| Topik mana yang **naik** dan kata apa yang **ramai**? | `analisis.tren_bulanan`, `analisis.tren_kata`, `analisis.kata_naik_turun` |
| Klaim baru ini **mirip hoaks lama yang mana**? | `analisis.cari_mirip`, `analisis.prediksi_klaster` |
| Bagaimana **bentuk** hoaks berubah (label, format, kategori)? | `muat_data.kelompokkan_label`, `teks.deteksi_format`, pivot di `app.py` |

---

## 2. Mengumpulkan data tanpa API

Turnbackhoax tidak menyediakan API publik untuk arsipnya, tetapi halaman daftar artikelnya
(`/articles?page=N`) adalah HTML biasa dengan pola yang rapi: setiap artikel adalah satu "kartu" berisi tautan,
gambar dengan atribut `alt` berisi judul lengkap, kategori, dan tanggal.

`scripts/kumpulkan_data.py` memakai `requests` untuk mengunduh dan **regex** untuk mengambil bagian yang dibutuhkan:

```python
POLA_KARTU = re.compile(r'<div class="news-card-h-alt.*?</div>\s*</div>\s*</div>', re.S)   # satu kartu
POLA_URL   = re.compile(r'href="(https://turnbackhoax\.id/articles/(\d+)-[^"]*)"')          # URL + id
POLA_JUDUL = re.compile(r'<img[^>]*alt="([^"]*)"')                                            # judul lengkap
```

Tiga kebiasaan yang perlu ditiru saat mengambil data dari situs orang:

1. **Pelan.** Tiga permintaan paralel, jeda setengah detik antar blok, dan pengulangan bertahap (1,5 s, 3 s, 4,5 s)
   bila server menjawab 5xx. Seluruh arsip (±1.500 halaman) selesai dalam 10-15 menit.
2. **Secukupnya.** Hanya metadata di halaman daftar yang disimpan; isi artikel tidak diunduh karena tidak dipakai.
3. **Bisa diperbarui.** Mode tanpa argumen hanya mengambil halaman terbaru sampai bertemu id yang sudah ada,
   sehingga pembaruan mingguan (GitHub Actions) hanya butuh beberapa permintaan.

Mengapa regex, bukan BeautifulSoup? Untuk pola yang sesederhana ini regex cukup dan tidak menambah dependensi.
Begitu struktur HTML berubah, keduanya sama-sama harus diperbaiki.

---

## 3. Membersihkan judul: lebih penting daripada algoritmanya

Komputer tidak membaca kata; ia menghitung. Sebelum dihitung, judul harus dibuat seragam. Lihat `src/teks.py`.

```python
"[SALAH] Video Prabowo Umumkan Pertalite Naik Jadi Rp13.000 – TurnBackHoax.ID"
 -> buang_label()   "Video Prabowo Umumkan Pertalite Naik Jadi Rp13.000"
 -> normalisasi()   "video prabowo umumkan pertalite naik jadi rp"
 -> tokenisasi()    ["prabowo", "umumkan", "pertalite"]
```

Tiga keputusan di balik pipeline itu:

**Awalan label dibuang dari teks, tetapi disimpan sebagai kolom sendiri.** `[SALAH]` bukan topik; ia adalah
vonis pemeriksa fakta. Kalau dibiarkan, kata "salah" akan menjadi fitur terbesar di setiap klaster.

**Dua lapis stopword.** Stopword umum (*yang, dan, akan*) ada di semua teks Indonesia. Stopword **domain**
(*hoaks, faktanya, video, foto, akun, beredar, facebook*) khas arsip cek fakta: ada di hampir semua judul, sehingga
tidak membantu membedakan topik. Coba hapus `STOPWORD_DOMAIN` dan lihat klaster "video" dan "foto" muncul.

**Tidak ada stemming.** Stemmer bahasa Indonesia tanpa kamus (memotong *me-, di-, -kan, -an*) sering merusak
kata yang justru penanda topik: *pemilu* → *ilu*, *menteri* → *teri*, *sekolah* → *kolah*. Yang dibuang hanya
akhiran *-nya* pada kata panjang (`rapikan_akhiran`). Latihan 6 mencoba Sastrawi (stemmer berbasis kamus) dan
membandingkan hasilnya.

Fungsi `deteksi_format` di file yang sama menebak bentuk unggahan dari kata di judul (video, foto, tautan, pesan
berantai). Ini heuristik kasar; cukup untuk melihat arah, bukan untuk angka pasti.

---

## 4. TF-IDF: mengubah judul menjadi vektor

Setelah bersih, setiap judul menjadi deretan angka sepanjang jumlah kosakata (puluhan ribu). Hampir semua angka
nol; hanya kata yang muncul di judul itu yang bernilai. Nilainya adalah **TF-IDF**:

$$\text{tfidf}(k, d) = (1 + \log \text{tf}_{k,d}) \times \left(\log\frac{1 + N}{1 + \text{df}_k} + 1\right)$$

- `tf` (*term frequency*): berapa kali kata *k* ada di judul *d*. Dengan `sublinear_tf=True`, 3 kemunculan
  tidak bernilai 3x, melainkan 1 + log 3 ≈ 2,1.
- `df` (*document frequency*): berapa judul yang memuat kata *k*. Semakin banyak, semakin kecil bobotnya.
- *N*: jumlah judul.

Intuisi: kata yang sering muncul di judul ini **dan** jarang di judul lain adalah kata yang paling menceritakan
judul ini. "prabowo" di satu judul bernilai tinggi; "indonesia" yang ada di ribuan judul bernilai rendah.

```python
vec = TfidfVectorizer(tokenizer=str.split, ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True)
X = vec.fit_transform(df["teks_tfidf"])      # bentuk: (jumlah judul, jumlah fitur), matriks jarang
```

| Parameter | Arti | Mengapa nilainya begitu |
|---|---|---|
| `ngram_range=(1, 2)` | kata tunggal + pasangan kata berurutan | "kartu prakerja", "anies baswedan" lebih bermakna daripada kata terpisahnya |
| `min_df=3` | buang fitur yang ada di < 3 judul | salah ketik dan nama yang muncul sekali hanya menambah dimensi |
| `max_df=0.5` | buang fitur yang ada di > 50% judul | hampir pasti kata umum yang lolos stopword |
| `tokenizer=str.split` | teks sudah bersih | pembersihan dilakukan sekali di `muat_data.siapkan`, bukan setiap kali model dibangun |

TfidfVectorizer menormalkan setiap baris ke panjang 1 (norma L2). Ini penting untuk dua bagian berikutnya.

---

## 5. Menemukan topik tanpa label: NMF dan K-Means

Keduanya algoritma **tak terawasi**: tidak ada contoh "ini topik A, itu topik B". Keduanya hanya diberi matriks
TF-IDF dan angka *k*. Aplikasi memakai **NMF** sebagai bawaan dan menyediakan K-Means sebagai pembanding di sidebar.

**NMF (Non-negative Matrix Factorization)** memecah matriks *X* (judul × kata) menjadi dua matriks kecil yang
semuanya non-negatif:

$$X \approx W \cdot H, \qquad W: \text{judul} \times k, \quad H: k \times \text{kata}$$

Baris *H* adalah topik: kata dengan bobot terbesar di baris itu menjadi kata kuncinya. Baris *W* adalah "resep"
setiap judul, seberapa banyak ia mengandung tiap topik. Label judul = topik dengan bobot terbesar.

```python
nmf = NMF(n_components=k, init="nndsvda", random_state=42).fit(X)
W = nmf.transform(X)          # judul x topik
label = W.argmax(axis=1)      # topik dominan per judul
H = nmf.components_           # topik x kata -> kata kunci
```

Karena semua nilai non-negatif, hasilnya mudah dibaca: topik adalah *penjumlahan* kata, bukan kombinasi dengan
tanda minus seperti pada PCA/LSA. `init="nndsvda"` memberi titik awal deterministik, sehingga hasil tidak berubah
tiap kali dijalankan.

**K-Means** mencari *k* titik pusat sehingga jarak kuadrat setiap judul ke pusat terdekatnya sekecil mungkin:

1. Pilih *k* pusat awal (scikit-learn memakai k-means++, yang menyebar pusat awal agar tidak menumpuk).
2. Tempelkan setiap judul ke pusat terdekat.
3. Geser setiap pusat ke rata-rata anggotanya.
4. Ulangi 2-3 sampai anggota tidak berpindah lagi.

```python
km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(X)
km.labels_            # nomor klaster setiap judul
km.cluster_centers_   # k vektor pusat, dimensi sama dengan X
```

**Memberi nama klaster.** Pusat klaster adalah rata-rata vektor anggotanya, jadi koordinat terbesarnya adalah kata
yang paling khas di klaster itu. `analisis.kata_kunci_klaster` mengurutkannya; `analisis.beri_nama` mengambil tiga
teratas yang tidak saling tumpang tindih ("vaksin covid" dan "vaksin" dianggap sama) dan menyambungnya dengan " · ".

**Memilih k.** Tidak ada jawaban benar. Terlalu kecil: Politik dan Kesehatan tercampur. Terlalu besar: satu topik
terpecah menjadi "gempa cianjur", "gempa turki", "gempa megathrust". Slider di sidebar ada supaya kamu bisa
merasakannya sendiri; 10-14 biasanya nyaman untuk arsip ini. Latihan 5 menghitung *silhouette score* sebagai
pembanding yang lebih objektif.

**Mengapa NMF menjadi bawaan.** Pada percobaan dengan seluruh arsip (±14.700 judul, k = 12), K-Means menumpuk
sekitar 70% judul di satu klaster "keranjang sisa" dan menyisakan topik-topik kecil; NMF menghasilkan topik yang
jauh lebih seimbang (topik terbesar ±30%) dengan kata kunci yang lebih koheren (Prabowo-Gibran, Jokowi-ijazah,
vaksin-Covid, undian berhadiah, lowongan kerja, dan seterusnya). Penyebabnya: K-Means memaksa setiap judul masuk
tepat satu klaster berdasarkan jarak, padahal judul pendek dengan 3-5 kata sering tidak dekat dengan pusat mana pun.
Judul yang menyinggung dua topik (vaksin + politik) pun hanya boleh masuk satu. NMF memberi *campuran* topik per
dokumen, dan argmax-nya cukup untuk label. Latihan 7 memperbaiki K-Means dengan LSA; Latihan 5 membandingkan
keduanya dengan silhouette.

---

## 6. Tren: dari klaster ke garis waktu

Begitu setiap judul punya nomor klaster, tren hanyalah agregasi pandas:

```python
pv = df.pivot_table(index="bulan", columns="topik", values="id", aggfunc="count", fill_value=0)
pv = pv.reindex(pd.date_range(pv.index.min(), pv.index.max(), freq="MS"), fill_value=0)   # bulan kosong = 0
```

Dua jebakan yang dihindari `analisis.py`:

- **Bulan tanpa artikel hilang dari pivot.** `reindex` dengan `date_range` mengisi nol, supaya garis tidak melompat.
- **Jumlah mentah menipu.** Tahun 2020 punya jauh lebih banyak artikel daripada 2017 karena tim pemeriksa fakta
  membesar, bukan karena hoaks tiga kali lipat. Karena itu tab Topic clusters dan Modus menampilkan **porsi**
  (persen dari artikel tahun itu), bukan jumlah.

**Kata yang naik.** `kata_naik_turun` membandingkan laju kemunculan kata per 1.000 judul sebelum dan sesudah tanggal
batas, dengan pemulusan Laplace:

$$\text{rasio} = \frac{(f_{\text{sesudah}} + 1)/(N_{\text{sesudah}} + 1)}{(f_{\text{sebelum}} + 1)/(N_{\text{sebelum}} + 1)}$$

Tanpa +1, kata yang baru lahir (misalnya nama peristiwa baru) dibagi nol. Tanpa ambang `min_total=8`, kata yang
muncul tiga kali tampil sebagai "naik 300%". Skala log pada sumbu x membuat kenaikan 10x dan penurunan 10x
sama panjangnya.

---

## 7. Kemiripan: mencari hoaks lama untuk klaim baru

Klaim yang kamu tempel di tab Similarity check diproses dengan pipeline **yang sama** dan diubah ke vektor dengan
kosakata model yang sama (`vectorizer.transform`, bukan `fit_transform`: kosakata tidak boleh berubah).
Lalu dibandingkan dengan semua judul arsip memakai *cosine similarity*:

$$\cos(\theta) = \frac{\vec a \cdot \vec b}{\lVert \vec a \rVert \lVert \vec b \rVert}$$

Karena semua vektor sudah berpanjang 1, penyebutnya 1, dan cosine = perkalian titik. Satu perkalian matriks jarang
menghasilkan skor untuk seluruh arsip:

```python
v = model.vectorizer.transform([teks.bersihkan_judul(klaim)])
skor = (model.X @ v.T).toarray().ravel()   # satu angka per judul arsip
urut = np.argsort(skor)[::-1][:n]
```

Yang diukur hanya **kesamaan kata**. "Harga BBM naik" dan "BBM makin mahal" dianggap tidak mirip karena tidak
berbagi kata. Itulah batas TF-IDF; *sentence embedding* (Latihan 8) memetakan makna, bukan kata, tetapi butuh
model berukuran ratusan MB.

Perhatikan juga apa yang **tidak** dilakukan fitur ini: ia tidak menyatakan klaim itu salah. Ia hanya menunjukkan
pemeriksaan fakta mana yang kata-katanya paling mirip, supaya orang bisa membacanya sendiri.

---

## 8. Menyusun aplikasi Streamlit yang tidak lambat

Melatih TF-IDF + K-Means pada belasan ribu judul butuh beberapa detik. Streamlit menjalankan ulang seluruh skrip
setiap kali ada interaksi, jadi tanpa cache setiap klik akan menunggu beberapa detik.

```python
@st.cache_data                      # untuk tabel: hasil disalin, aman diubah
def muat_arsip(): ...

@st.cache_resource                  # untuk objek berat: model sklearn, satu instance dipakai bersama
def bangun_model(ids: tuple, dokumen: tuple, k: int): ...
```

Kunci cache `bangun_model` adalah `(ids, dokumen, k)`: model hanya dilatih ulang bila filter (tahun/kategori) atau
*k* berubah. Argumen harus *hashable*, karena itu dikirim sebagai `tuple`, bukan DataFrame.

Pola lain yang dipakai di `app.py`:

- Semua warna didefinisikan sekali (`WARNA_LABEL`, `WARNA_FORMAT`, `PALET`) dan dipakai lewat `color_discrete_map`
  agar grafik di tab mana pun konsisten.
- `rapikan()` memberi gaya seragam pada semua figur Plotly.
- Learning mode hanyalah `if mode_belajar: with st.expander(...)` di akhir setiap tab, sehingga pengguna biasa
  tidak terganggu dan pembelajar bisa membuka rumusnya.
- RSS diambil lewat fungsi ber-cache 30 menit dengan kunci waktu, supaya tidak memukul server setiap rerun.

---

## 9. Pengujian

`tests/` berisi 27 tes yang berjalan tanpa internet dalam beberapa detik:

- `test_teks.py`: pembersihan judul, stopword, deteksi format.
- `test_muat_data.py`: pengelompokan label, kolom turunan, parsing RSS dari string XML, dan satu tes yang membaca
  arsip di repo (jumlah baris, id unik, rentang tahun).
- `test_analisis.py`: korpus kecil buatan dengan dua topik yang jelas (vaksin vs gempa). Tes dijalankan untuk NMF
  **dan** K-Means (`pytest.fixture(params=...)`) dan memastikan keduanya memisahkan dua topik itu, nama topik memuat
  kata kuncinya, tren mengisi bulan kosong, dan pencarian kemiripan mengembalikan judul dari topik yang benar.

Mengapa korpus buatan? Karena hasil K-Means pada data nyata bisa berubah tiap kali data diperbarui; tes harus
menguji **perilaku** fungsi, bukan angka tertentu dari satu snapshot.

---

## 10. Istilah

| Istilah | Arti singkat |
|---|---|
| Bag of words | Representasi teks sebagai kantong kata tanpa urutan |
| TF-IDF | Bobot kata: sering di dokumen ini, jarang di dokumen lain |
| N-gram | Deretan *n* kata berurutan; bigram = pasangan kata |
| Stopword | Kata yang dibuang karena tidak membedakan dokumen |
| Stemming | Memotong imbuhan ke kata dasar; tidak dipakai di sini |
| Matriks jarang (sparse) | Matriks yang hampir seluruhnya nol, disimpan hemat memori |
| NMF | Faktorisasi matriks non-negatif X ≈ W·H; baris H = topik, baris W = campuran topik per dokumen |
| K-Means | Klasterisasi tak terawasi dengan *k* pusat |
| Pusat klaster (centroid) | Rata-rata vektor anggota satu klaster |
| Cosine similarity | Kemiripan arah dua vektor, 0 sampai 1 untuk TF-IDF |
| Laju per 1.000 | Frekuensi dinormalkan terhadap jumlah dokumen, untuk membandingkan periode berbeda |
| Pemulusan Laplace | Menambah 1 pada hitungan agar tidak ada pembagian nol |
