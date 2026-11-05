"""
src/teks.py - Pembersihan teks bahasa Indonesia sebelum masuk TF-IDF
====================================================================
Semua fungsi di sini murni (tanpa Streamlit, tanpa I/O) supaya mudah diuji.

Alur: judul mentah -> buang awalan label "[SALAH]" -> huruf kecil -> buang URL, angka, tanda baca
      -> pecah jadi kata -> buang stopword umum + stopword domain -> rapikan akhiran "-nya".

Mengapa tidak memakai stemmer (Sastrawi dan sebagainya)? Stemmer tanpa kamus memotong kata seperti
"pemilu" atau "menteri" secara keliru, padahal nama dan istilah seperti itu justru penanda topik yang
paling penting. Untuk tujuan klasterisasi judul, stopword yang baik lebih berpengaruh daripada stemming.
Latihan di docs/LATIHAN.md mengajak mencoba Sastrawi dan membandingkan hasilnya.
"""

from __future__ import annotations

import re

# Stopword bahasa Indonesia: kata fungsi yang hampir selalu muncul dan tidak menandai topik.
STOPWORD_UMUM = set("""
ada adalah adanya adapun agak agar akan akankah akhir akhiri akhirnya aku akulah amat amatlah anda andalah antar antara
antaranya apa apaan apabila apakah apalagi apatah artinya asal asalkan atas atau ataukah ataupun awal awalnya bagai bagaikan
bagaimana bagaimanakah bagaimanapun bagi bagian bahkan bahwa bahwasanya baik bakal bakalan balik banyak bapak baru bawah
beberapa begini beginian beginikah beginilah begitu begitukah begitulah begitupun bekerja belakang belakangan belum
belumlah benar benarkah benarlah berada berakhir berakhirlah berakhirnya berapa berapakah berapalah berapapun berarti
berawal berbagai berdatangan beri berikan berikut berikutnya berjumlah berkali berkata berkehendak berkeinginan
berkenaan berlainan berlalu berlangsung berlebihan bermacam bermaksud bermula bersama bersiap bersiap-siap bertanya
bertanya-tanya berturut berturut-turut bertutur berujar berupa besar betul betulkah biasa biasanya bila bilakah bisa
bisakah boleh bolehkah bolehlah buat bukan bukankah bukanlah bukannya bulan bung cara caranya cukup cukupkah cukuplah
cuma dahulu dalam dan dapat dari daripada datang dekat demi demikian demikianlah dengan depan di dia diakhiri diakhirinya
dialah diantara diantaranya diberi diberikan diberikannya dibuat dibuatnya didapat didatangkan digunakan diibaratkan
diibaratkannya diingat diingatkan diinginkan dijawab dijelaskan dijelaskannya dikarenakan dikatakan dikatakannya
dikerjakan diketahui diketahuinya dikira dilakukan dilalui dilihat dimaksud dimaksudkan dimaksudkannya dimaksudnya
diminta dimintai dimisalkan dimulai dimulailah dimulainya dimungkinkan dini dipastikan diperbuat diperbuatnya
dipergunakan diperkirakan diperlihatkan diperlukan diperlukannya dipersoalkan dipertanyakan dipunyai diri dirinya
disampaikan disebut disebutkan disebutkannya disini disinilah ditambahkan ditandaskan ditanya ditanyai ditanyakan
ditegaskan ditujukan ditunjuk ditunjuki ditunjukkan ditunjukkannya ditunjuknya dituturkan dituturkannya diucapkan
diucapkannya diungkapkan dong dua dulu empat enggak enggaknya entah entahlah guna gunakan hal hampir hanya hanyalah
hari harus haruslah harusnya hendak hendaklah hendaknya hingga ia ialah ibarat ibaratkan ibaratnya ibu ikut ingat
ingat-ingat ingin inginkah inginkan ini inikah inilah itu itukah itulah jadi jadilah jadinya jangan jangankan janganlah
jauh jawab jawaban jawabnya jelas jelaskan jelaslah jelasnya jika jikalau juga jumlah jumlahnya justru kala kalau
kalaulah kalaupun kalian kami kamilah kamu kamulah kan kapan kapankah kapanpun karena karenanya kasus kata katakan
katakanlah katanya ke keadaan kebetulan kecil kedua keduanya keinginan kelamaan kelihatan kelihatannya kelima keluar
kembali kemudian kemungkinan kemungkinannya kenapa kepada kepadanya kesampaian keseluruhan keseluruhannya keterlaluan
ketika khususnya kini kinilah kira kira-kira kiranya kita kitalah kok kurang lagi lagian lah lain lainnya lalu lama
lamanya lanjut lanjutnya lebih lewat lima luar macam maka makanya makin malah malahan mampu mampukah mana manakala
manalagi masa masalah masalahnya masih masihkah masing masing-masing mau maupun melainkan melakukan melalui melihat
melihatnya memang memastikan memberi memberikan membuat memerlukan memihak meminta memintakan memisalkan memperbuat
mempergunakan memperkirakan memperlihatkan mempersiapkan mempersoalkan mempertanyakan mempunyai memulai memungkinkan
menaiki menambahkan menandaskan menanti menanti-nanti menantikan menanya menanyai menanyakan mendapat mendapatkan
mendatang mendatangi mendatangkan menegaskan mengakhiri mengapa mengatakan mengatakannya mengenai mengerjakan mengetahui
menggunakan menghendaki mengibaratkan mengibaratkannya mengingat mengingatkan menginginkan mengira mengucapkan
mengucapkannya mengungkapkan menjadi menjawab menjelaskan menuju menunjuk menunjuki menunjukkan menunjuknya menurut
menuturkan menyampaikan menyangkut menyatakan menyebutkan menyeluruh menyiapkan merasa mereka merekalah merupakan
meski meskipun meyakini meyakinkan minta mirip misal misalkan misalnya mula mulai mulailah mulanya mungkin mungkinkah
nah naik namun nanti nantinya nyaris nyatanya oleh olehnya pada padahal padanya pak paling panjang pantas para pasti
pastilah penting pentingnya per percuma perlu perlukah perlunya pernah persoalan pertama pertama-tama pertanyaan
pertanyakan pihak pihaknya pukul pula pun punya rasa rasanya rata rupanya saat saatnya saja sajalah saling sama
sama-sama sambil sampai sampai-sampai sampaikan sana sangat sangatlah satu saya sayalah se sebab sebabnya sebagai
sebagaimana sebagainya sebagian sebaik sebaik-baiknya sebaiknya sebaliknya sebanyak sebegini sebegitu sebelum sebelumnya
sebenarnya seberapa sebesar sebetulnya sebisanya sebuah sebut sebutlah sebutnya secara secukupnya sedang sedangkan
sedemikian sedikit sedikitnya seenaknya segala segalanya segera seharusnya sehingga seingat sejak sejauh sejenak
sejumlah sekadar sekadarnya sekali sekali-kali sekalian sekaligus sekalipun sekarang sekarang sekecil seketika sekiranya
sekitar sekitarnya sekurang-kurangnya sekurangnya sela selagi selain selaku selalu selama selama-lamanya selamanya
selanjutnya seluruh seluruhnya semacam semakin semampu semampunya semasa semasih semata semata-mata semaunya sementara
semisal semisalnya sempat semua semuanya semula sendiri sendirian sendirinya seolah seolah-olah seorang sepanjang
sepantasnya sepantasnyalah seperlunya seperti sepertinya sepihak sering seringnya serta serupa sesaat sesama sesampai
sesegera sesekali seseorang sesuatu sesuatunya sesudah sesudahnya setelah setempat setengah seterusnya setiap setiba
setibanya setidak-tidaknya setidaknya setinggi seusai sewaktu siap siapa siapakah siapapun sini sinilah soal soalnya
suatu sudah sudahkah sudahlah supaya tadi tadinya tahu tahun tak tambah tambahnya tampak tampaknya tandas tandasnya
tanpa tanya tanyakan tanyanya tapi tegas tegasnya telah tempat tengah tentang tentu tentulah tentunya tepat terakhir
terasa terbanyak terdahulu terdapat terdiri terhadap terhadapnya teringat teringat-ingat terjadi terjadilah terjadinya
terkira terlalu terlebih terlihat termasuk ternyata tersampaikan tersebut tersebutlah tertentu tertuju terus terutama
tetap tetapi tiap tiba tiba-tiba tidak tidakkah tidaklah tiga tinggi toh tunjuk turut tutur tuturnya ucap ucapnya ujar
ujarnya umum umumnya ungkap ungkapnya untuk usah usai waduh wah wahai waktu waktunya walau walaupun wong yaitu yakin
yakni yang
""".split())

# Stopword domain: kata yang muncul di hampir semua judul cek fakta sehingga tidak membedakan topik.
STOPWORD_DOMAIN = set("""
hoaks hoax faktanya fakta cek klaim beredar unggahan unggah akun narasi informasi kabar berita artikel pesan
tbh turnbackhoax mafindo salah benar klarifikasi disinformasi misinformasi penipuan
video foto gambar tautan link situs website konten postingan post viral media sosial medsos
facebook twitter instagram tiktok whatsapp youtube telegram wa fb ig x
diklaim mengklaim disebut menyebut katanya tersebar menyebar sebar
""".split())

STOPWORD = STOPWORD_UMUM | STOPWORD_DOMAIN

# Kata yang menandai format unggahan; dianalisis terpisah (tab "Modus"), bukan untuk topik.
KATA_FORMAT = {
    "Video": ("video", "rekaman", "cuplikan", "siaran"),
    "Foto / gambar": ("foto", "gambar", "tangkapan layar", "screenshot", "poster", "meme"),
    "Tautan / situs": ("tautan", "link", "situs", "website", "url", "pendaftaran", "formulir"),
    "Pesan berantai": ("pesan berantai", "pesan", "broadcast", "whatsapp", "wa", "sms"),
}

POLA_LABEL = re.compile(r"^\s*[\[\(]\s*[^\]\)]*?\s*[\]\)]\s*")
POLA_URL = re.compile(r"https?://\S+|www\.\S+")
POLA_NON_HURUF = re.compile(r"[^a-z\s]")
POLA_SPASI = re.compile(r"\s+")


def buang_label(judul: str) -> str:
    """'[SALAH] Vaksin Mengandung Chip' -> 'Vaksin Mengandung Chip'. Juga membuang akhiran situs."""
    judul = POLA_LABEL.sub("", judul or "")
    judul = re.sub(r"\s*[–-]\s*TurnBackHoax\.ID\s*$", "", judul, flags=re.I)
    return judul.strip().strip('"“”\'')


def normalisasi(teks: str) -> str:
    """Huruf kecil, tanpa URL, angka, dan tanda baca. Tanda kutip ganda yang aneh ikut hilang."""
    teks = (teks or "").lower()
    teks = POLA_URL.sub(" ", teks)
    teks = teks.replace("-", " ")
    teks = POLA_NON_HURUF.sub(" ", teks)
    return POLA_SPASI.sub(" ", teks).strip()


def rapikan_akhiran(kata: str) -> str:
    """Hanya akhiran -nya yang dibuang (aman): 'videonya' -> 'video'. Tanpa stemming lain."""
    if len(kata) > 5 and kata.endswith("nya"):
        return kata[:-3]
    return kata


def tokenisasi(teks: str, stopword: set[str] = STOPWORD, min_panjang: int = 3) -> list[str]:
    """Pecah teks menjadi daftar kata bermakna. Dipakai sebagai `tokenizer` TfidfVectorizer."""
    hasil = []
    for kata in normalisasi(teks).split():
        kata = rapikan_akhiran(kata)
        if len(kata) >= min_panjang and kata not in stopword:
            hasil.append(kata)
    return hasil


def bersihkan_judul(judul: str) -> str:
    """Pipeline lengkap untuk satu judul; hasilnya string kata-kata bersih yang dipisah spasi."""
    return " ".join(tokenisasi(buang_label(judul)))


def deteksi_format(judul: str) -> str:
    """Tebak bentuk unggahan dari kata di judul. Urutan dict menentukan prioritas bila lebih dari satu cocok."""
    j = normalisasi(buang_label(judul))
    kata = set(j.split())
    for nama, penanda in KATA_FORMAT.items():
        for p in penanda:
            if (" " in p and p in j) or p in kata:
                return nama
    return "Teks / lainnya"
