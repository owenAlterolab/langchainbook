# File konfigurasi untuk mengelola API KEY sebagai variabel lingkungan
from dotenv import load_dotenv
from langchain.memory import ConversationTokenBufferMemory
from langchain_openai import ChatOpenAI

# Memuat informasi API KEY
load_dotenv()

# membuat model LLM
llm = ChatOpenAI()

# Mengatur memori
memory = ConversationTokenBufferMemory(
    llm = llm, max_token_limit = 150, return_messages = True # batasi panjang token maksimum hingga 150
)

memory.save_context(
    inputs={
        "human": "Halo, saya baru saja membeli mesin dari perusahaan Anda. Bisakah Anda memberitahu saya cara memasangnya?"
    },
    outputs={
        "ai": "Halo! Terima kasih telah melakukan pembelian. Bisa Anda berikan nomor model mesin tersebut?"
    },
)
memory.save_context(
    inputs={"human": "Ya, nomor modelnya adalah XG-200."},
    outputs={
        "ai": "Terima kasih. Saya akan membantu Anda dengan panduan instalasi model XG-200. Pertama, tolong pastikan keadaan suplai daya di lokasi instalasi. Mesin ini memerlukan daya 220V."
    },
)
memory.save_context(
    inputs={"human": "Saya sudah memastikan suplai daya. Apa langkah selanjutnya?"},
    outputs={
        "ai": "Bagus. Selanjutnya, letakkan mesin di lantai yang rata dan stabil. Setelah itu, silakan lanjutkan dengan menyambungkan kabel sesuai dengan panduan pengguna yang disediakan."
    },
)
memory.save_context(
    inputs={"human": "Bagaimana cara menyambungkan kabelnya?"},
    outputs={
        "ai": "Silakan merujuk ke halaman 5 dari panduan pengguna. Di sana terdapat instruksi lengkap tentang penyambungan kabel. Jika Anda mengalami kesulitan, kami siap membantu lebih lanjut."
    },
)
memory.save_context(
    inputs={"human": "Apa yang harus saya lakukan setelah instalasi selesai?"},
    outputs={
        "ai": "Setelah instalasi selesai, nyalakan mesin dan lakukan pengujian awal. Prosedur pengujian dijelaskan di halaman 10 dari panduan pengguna. Jika ada masalah dengan mesin atau Anda memerlukan bantuan tambahan, silakan hubungi kami kapan saja."
    },
)
memory.save_context(
    inputs={"human": "Terima kasih, ini sangat membantu!"},
    outputs={
        "ai": "Kami selalu siap untuk membantu. Jika Anda memiliki pertanyaan lain atau memerlukan dukungan tambahan, jangan ragu untuk menghubungi kami. Semoga harimu menyenangkan!"
    },
)

# Memeriksa riwayat percakapan.
print(memory.load_memory_variables({})["history"])