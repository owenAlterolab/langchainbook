# File konfigurasi untuk mengelola API KEY sebagai variabel lingkungan
from dotenv import load_dotenv
from langchain.memory import ConversationSummaryMemory, ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI


# Memuat informasi API KEY
load_dotenv()

memory = ConversationSummaryMemory(llm=ChatOpenAI(temperature=0), return_messages=True)

memory.save_context(
    inputs={"human": "Berapa harga paket perjalanan ke Eropa?"},
    outputs={
        "ai": "Harga dasar untuk paket Eropa 14 hari 15 malam adalah 3.500 euro. Harga ini sudah termasuk biaya penerbangan, akomodasi hotel, dan biaya masuk ke objek wisata yang ditentukan. Biaya tambahan dapat bervariasi tergantung pada tur opsional yang Anda pilih atau biaya pribadi."
    },
)
memory.save_context(
    inputs={"human": "Apa saja tempat wisata utama yang akan kami kunjungi selama perjalanan?"},
    outputs={
        "ai": "Perjalanan ini mencakup kunjungan ke landmark terkenal di Eropa seperti Menara Eiffel di Paris, Colosseum di Roma, Gerbang Brandenburg di Berlin, dan Air Terjun Rhine di Zurich. Anda akan dapat menikmati pemandangan ikonik dari setiap kota secara menyeluruh."
    },
)
memory.save_context(
    inputs={"human": "Apakah asuransi perjalanan sudah termasuk?"},
    outputs={
        "ai": "Ya, asuransi perjalanan dasar disediakan untuk semua wisatawan. Asuransi ini mencakup biaya medis, bantuan darurat, dan lainnya. Jika Anda menginginkan perlindungan asuransi tambahan, Anda dapat meningkatkan paket Anda."
    },
)
memory.save_context(
    inputs={
        "human": "Bisakah saya meng-upgrade kursi penerbangan saya ke kelas bisnis? Berapa biayanya?"
    },
    outputs={
        "ai": "Meng-upgrade kursi penerbangan Anda ke kelas bisnis adalah mungkin. Biaya upgrade sekitar 1.200 euro untuk pulang-pergi. Kelas bisnis menawarkan manfaat seperti kursi yang lebih lebar, makanan dalam penerbangan yang lebih baik, dan tambahan jatah bagasi."
    },
)
memory.save_context(
    inputs={"human": "Apa tingkat hotel yang termasuk dalam paket?"},
    outputs={
        "ai": "Paket ini mencakup akomodasi di hotel bintang 4. Setiap hotel menyediakan kenyamanan dan kemudahan, dengan lokasi yang memudahkan akses ke objek wisata. Semua hotel dilengkapi dengan layanan dan fasilitas yang luar biasa."
    },
)
memory.save_context(
    inputs={"human": "Bisakah Anda memberikan lebih banyak detail tentang pilihan makanan?"},
    outputs={
        "ai": "Paket perjalanan ini termasuk sarapan setiap hari yang disediakan oleh hotel. Makan siang dan makan malam tidak termasuk, memungkinkan wisatawan untuk menikmati berbagai makanan lokal. Selain itu, kami menyediakan daftar restoran yang direkomendasikan di setiap kota untuk membantu Anda menikmati kuliner lokal sepenuhnya."
    },
)
memory.save_context(
    inputs={"human": "Berapa deposit yang diperlukan saat memesan paket? Apa kebijakan pembatalannya?"},
    outputs={
        "ai": "Deposit sebesar 500 euro diperlukan saat memesan paket. Kebijakan pembatalan memungkinkan pengembalian dana penuh hingga 30 hari sebelum tanggal pemesanan, setelah itu deposit tidak dapat dikembalikan. Jika Anda membatalkan dalam 14 hari sebelum tanggal keberangkatan, 50% dari biaya akan dikenakan, dan biaya penuh akan dikenakan setelah itu."
    },
)

# print(memory.load_memory_variables({})["history"])

llm = ChatOpenAI()
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=200,  # Mengatur panjang token yang menjadi dasar ringkasan.
    return_messages=True,
)

memory.save_context(
    inputs={"human": "Apa saja tempat wisata utama yang akan dikunjungi selama perjalanan?"},
    outputs={
        "ai": "Dalam perjalanan ini, Anda akan mengunjungi tempat-tempat wisata terkenal di Eropa seperti Menara Eiffel di Paris, Colosseum di Roma, Gerbang Brandenburg di Berlin, dan Air Terjun Rhine di Zurich. Anda akan dapat menikmati pemandangan ikonik dari setiap kota secara menyeluruh."
    },
)
memory.save_context(
    inputs={"human": "Apakah asuransi perjalanan sudah termasuk?"},
    outputs={
        "ai": "Ya, semua wisatawan mendapatkan asuransi perjalanan dasar. Asuransi ini mencakup dukungan untuk biaya medis dan bantuan dalam situasi darurat. Jika Anda menginginkan perlindungan asuransi tambahan, peningkatan paket tersedia."
    },
)
memory.save_context(
    inputs={
        "human": "Bisakah saya meng-upgrade kursi penerbangan saya ke kelas bisnis? Berapa biayanya?"
    },
    outputs={
        "ai": "Ya, Anda dapat meng-upgrade kursi penerbangan Anda ke kelas bisnis. Biaya upgrade sekitar 1.200 euro untuk perjalanan pulang-pergi. Kelas bisnis menawarkan kursi yang lebih luas, makanan dalam penerbangan yang lebih baik, dan tambahan jatah bagasi."
    },
)
memory.save_context(
    inputs={"human": "Apa tingkat hotel yang termasuk dalam paket ini?"},
    outputs={
        "ai": "Paket ini mencakup akomodasi di hotel bintang 4. Setiap hotel menawarkan kenyamanan dan kemudahan, dengan lokasi strategis yang memudahkan akses ke tempat wisata. Semua hotel dilengkapi dengan layanan dan fasilitas yang unggul."
    },
)

print(memory.load_memory_variables({})["history"])