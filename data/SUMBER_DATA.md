# Sumber data

Semua data dalam proyek ini publik dan dapat dilihat siapa pun di browser. Tidak ada yang berasal dari akun berbayar,
API tertutup, atau pengambilan konten yang dilindungi.

## 1. Arsip artikel cek fakta — Turnbackhoax (Mafindo)

- **Apa:** metadata artikel cek fakta yang diterbitkan Masyarakat Anti Fitnah Indonesia (Mafindo) di
  [turnbackhoax.id](https://turnbackhoax.id): judul (dengan awalan label seperti `[SALAH]`, `[PENIPUAN]`,
  `[KLARIFIKASI]`), kategori (Politik, Kesehatan, Bencana, ...), tanggal terbit, satu kalimat ringkasan
  ("Faktanya, ...") yang terpotong seperti tampil di halaman daftar, dan URL artikel.
- **Dari mana:** halaman daftar artikel publik `https://turnbackhoax.id/articles?page=N`, 10 kartu per halaman.
  `scripts/kumpulkan_data.py` membaca kartu-kartu itu dan menyimpannya ke `hoaks_turnbackhoax.csv`.
  Isi artikel, gambar, dan uraian pemeriksa fakta **tidak** diunduh.
- **Cakupan:** seluruh halaman daftar yang tersedia pada tanggal pengambilan (tertulis di README dan di sidebar
  aplikasi). Artikel lama sebelum situs Turnbackhoax dimigrasikan ke platform barunya sebagian tidak lagi muncul
  di daftar, sehingga tahun-tahun awal lebih tipis daripada jumlah sebenarnya.
- **Pembaruan:** `python scripts/kumpulkan_data.py` (tanpa argumen) mengambil halaman terbaru sampai bertemu
  artikel yang sudah tersimpan. Workflow GitHub Actions `perbarui-data.yml` menjalankannya setiap minggu.
  `--semua` mengunduh ulang seluruh arsip (±10-15 menit, 3 permintaan paralel dengan jeda).
- **Kolom:**

  | Kolom | Isi |
  |---|---|
  | `id` | nomor artikel dari URL (`/articles/37347-...`) |
  | `tanggal` | tanggal terbit, `YYYY-MM-DD` |
  | `judul` | judul lengkap persis seperti di situs, termasuk awalan label |
  | `label` | awalan label yang diekstrak dari judul (`SALAH`, `PENIPUAN`, `HOAX`, `KLARIFIKASI`, ...) |
  | `kategori` | kategori yang diberikan Turnbackhoax; kosong untuk artikel lama |
  | `ringkasan` | kalimat pembuka penjelasan, terpotong (±80 karakter) |
  | `url` | tautan artikel |

- **Lisensi dan atribusi:** konten Turnbackhoax adalah hak Mafindo. Proyek ini hanya menyimpan metadata ringkas
  (judul, label, kategori, tanggal, tautan) untuk analisis dan selalu menautkan kembali ke artikel aslinya.
  Jika Mafindo meminta data ini dihapus atau dibatasi, repo akan menyesuaikan.
- **Tanggal yang mustahil:** sekitar 6% kartu (±975 dari 15.664) tampil di situs dengan tahun seperti 1000 atau
  6325, atau tanpa tanggal, tampaknya akibat kesalahan migrasi data di sisi situs. Baris itu dibuang saat CSV
  disimpan (`kumpulkan_data.buang_tanggal_mustahil`; hanya 1 Januari 2014 sampai hari ini yang dipakai), dan
  `muat_data.siapkan` menerapkan saringan yang sama sebagai pengaman.
- **Keterbatasan:** (1) label dan kategori adalah keputusan pemeriksa fakta Mafindo, bukan penilaian proyek ini;
  (2) jumlah artikel per bulan mencerminkan kapasitas tim pemeriksa fakta, bukan jumlah hoaks yang sebenarnya
  beredar; (3) ringkasan terpotong dan hanya dipakai untuk tampilan, bukan analisis.

## 2. Artikel terbaru — RSS Turnbackhoax

- **URL:** `https://turnbackhoax.id/feed` (RSS 2.0 resmi, ±10 artikel terakhir).
- **Dipakai untuk:** bagian "Terbaru dari Turnbackhoax" di tab Radar. Setiap judul baru dipetakan ke klaster
  terdekat oleh model yang sama. Cache 30 menit; bila server tidak terjangkau, aplikasi menampilkan pemberitahuan
  dan tetap berjalan dari arsip.

## 3. Stopword — `src/teks.py`

- Daftar stopword bahasa Indonesia umum (±780 kata fungsi seperti *yang, dan, akan, sebagai*) disusun dari
  daftar-daftar yang lazim dipakai di komunitas NLP Indonesia, ditambah **stopword domain** (±60 kata yang muncul
  di hampir semua judul cek fakta: *hoaks, faktanya, video, foto, akun, beredar, klaim, unggahan*, nama platform).
- Mengapa stopword domain penting: tanpa itu, K-Means akan membentuk klaster "video", "foto", dan "akun"
  yang tidak bercerita apa pun tentang topik hoaksnya.

## Catatan etika

Metadata yang dikumpulkan adalah informasi yang memang dipublikasikan untuk dibaca publik, diambil dengan laju
rendah (3 permintaan paralel, jeda antar blok, pengulangan sopan saat server sibuk) dan hanya dari halaman daftar
yang disediakan untuk pengunjung. Proyek ini tidak menilai kebenaran klaim apa pun; label tetap milik pemeriksa
fakta. Tujuannya edukatif: menunjukkan bagaimana teknik NLP sederhana dapat merangkum ribuan laporan hoaks.
