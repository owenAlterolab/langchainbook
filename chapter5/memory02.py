from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2, return_messages=True)

memory.save_context(
    inputs={
        "human": "Halo, saya ingin membuka rekening bank secara daring. Bagaimana cara memulainya?"
    },
    outputs={
        "ai": "Halo! Kami senang mendengar Anda ingin membuka rekening. Pertama, bisakah Anda menyiapkan kartu identitas untuk verifikasi identitas?"
    },
)
memory.save_context(
    inputs={"human": "Ya, saya sudah menyiapkan kartu identitas. Apa yang harus saya lakukan selanjutnya?"},
    outputs={
        "ai": "Terima kasih. Silakan ambil foto yang jelas dari bagian depan dan belakang kartu identitas Anda dan unggah. Kami akan melanjutkan proses verifikasi identitas."
    },
)
memory.save_context(
    inputs={"human": "Saya telah mengunggah foto. Bagaimana cara melanjutkan verifikasi identitas?"},
    outputs={
        "ai": "Kami telah memverifikasi foto yang Anda unggah. Sekarang, silakan lanjutkan verifikasi identitas melalui ponsel Anda. Masukkan nomor verifikasi yang telah dikirimkan kepada Anda melalui SMS."
    },
)
memory.save_context(
    inputs={"human": "Saya sudah memasukkan nomor verifikasi. Bagaimana cara membuka rekening sekarang?"},
    outputs={
        "ai": "Verifikasi identitas Anda telah selesai. Sekarang, silakan pilih jenis rekening yang Anda inginkan dan masukkan informasi yang diperlukan. Anda dapat memilih jenis tabungan, jenis mata uang, dan lain-lain."
    },
)
memory.save_context(
    inputs={"human": "Saya sudah memasukkan semua informasi. Apa langkah selanjutnya?"},
    outputs={
        "ai": "Informasi yang Anda masukkan telah kami verifikasi. Proses pembukaan rekening hampir selesai. Terakhir, silakan setujui syarat dan ketentuan yang berlaku, dan konfirmasi pembukaan rekening."
    },
)
memory.save_context(
    inputs={"human": "Saya sudah menyelesaikan semua prosedur. Apakah rekening saya sudah dibuka?"},
    outputs={
        "ai": "Ya, pembukaan rekening telah selesai. Nomor rekening Anda dan informasi terkait telah dikirimkan ke email yang Anda daftarkan. Jika Anda memerlukan bantuan lebih lanjut, jangan ragu untuk menghubungi kami. Terima kasih!"
    },
)

print(memory.load_memory_variables({})["history"])