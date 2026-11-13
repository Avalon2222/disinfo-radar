"""Uji pembersihan teks (src/teks.py): tanpa internet, tanpa data."""

from src import teks


def test_buang_label_berbagai_gaya():
    assert teks.buang_label("[SALAH] Vaksin Mengandung Chip") == "Vaksin Mengandung Chip"
    assert teks.buang_label("(HOAX) Kerusuhan di Puncak, Bogor – TurnBackHoax.ID") == "Kerusuhan di Puncak, Bogor"
    assert teks.buang_label("[PENIPUAN] “Tautan Bansos” ") == "Tautan Bansos"
    assert teks.buang_label("Judul tanpa label") == "Judul tanpa label"
    assert teks.buang_label("") == ""


def test_normalisasi_membuang_url_angka_tanda_baca():
    hasil = teks.normalisasi("Pertalite Naik Rp13.000! cek https://contoh.id/x?y=1 (Jakarta-Bandung)")
    assert hasil == "pertalite naik rp cek jakarta bandung"


def test_tokenisasi_membuang_stopword_umum_dan_domain():
    token = teks.tokenisasi("Faktanya, video yang beredar menunjukkan vaksin COVID-19 mengandung chip")
    assert "faktanya" not in token and "video" not in token and "yang" not in token
    assert token == ["vaksin", "covid", "mengandung", "chip"]


def test_rapikan_akhiran_nya_saja():
    assert teks.rapikan_akhiran("videonya") == "video"
    assert teks.rapikan_akhiran("tanya") == "tanya"        # terlalu pendek, jangan dipotong
    assert teks.rapikan_akhiran("menteri") == "menteri"    # tidak ada stemming awalan


def test_bersihkan_judul_pipeline_lengkap():
    assert teks.bersihkan_judul("[SALAH] Video Prabowo Umumkan Pertalite Naik Jadi Rp13.000") == "prabowo umumkan pertalite"


def test_deteksi_format():
    assert teks.deteksi_format("[PENIPUAN] Tautan Bansos Pelajar Dari Kemensos") == "Tautan / situs"
    assert teks.deteksi_format("[SALAH] Foto Karhutla Gunung Lawu") == "Foto / gambar"
    assert teks.deteksi_format("[SALAH] Video Dapur MBG Dipantau WHO") == "Video"
    assert teks.deteksi_format("[SALAH] Pesan Berantai Bantuan Kuota") == "Pesan berantai"
    assert teks.deteksi_format("[SALAH] Dapur MBG Dipantau WHO") == "Teks / lainnya"


def test_input_kosong_atau_none_tidak_error():
    assert teks.buang_label(None) == ""
    assert teks.normalisasi(None) == ""
    assert teks.tokenisasi("") == []
    assert teks.bersihkan_judul("[SALAH]") == ""
    assert teks.deteksi_format("") == "Teks / lainnya"


def test_rapikan_akhiran_kata_pendek_dibiarkan():
    # "punya" dan "hanya" bukan kata + akhiran -nya, jadi tidak boleh dipotong.
    assert teks.rapikan_akhiran("punya") == "punya"
    assert teks.rapikan_akhiran("hanya") == "hanya"
