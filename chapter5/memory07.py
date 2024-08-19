# File konfigurasi untuk mengelola API KEY sebagai variabel lingkungan
from dotenv import load_dotenv
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory


# Memuat informasi API KEY
load_dotenv()

# Tentukan model
embeddings_model = OpenAIEmbeddings()

# Menginisialisasi Vektor penyimpanan.
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})

# Ini untuk menunjukkan bahwa pengambilan vektor masih mengembalikan informasi yang relevan secara semantik.
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
memory = VectorStoreRetrieverMemory(retriever=retriever)

# Menyimpan percakapan acak.
memory.save_context(
    inputs={
        "human": "Halo, terima kasih telah menghadiri wawancara hari ini. Bisakah Anda memperkenalkan diri?"
    },
    outputs={
        "ai": "Halo. Saya adalah seorang pengembang baru yang memiliki latar belakang di ilmu komputer. Di universitas, saya terutama menggunakan Java dan Python, dan baru-baru ini saya berpartisipasi dalam proyek pengembangan web di mana saya mendapatkan pengalaman dalam mengembangkan layanan untuk pengguna sebenarnya."
    },
)
memory.save_context(
    inputs={"human": "Apa peran Anda dalam proyek tersebut?"},
    outputs={
        "ai": "Peran saya adalah sebagai pengembang backend. Saya bertanggung jawab untuk mengelola data pengguna dan mengembangkan logika server, serta mengimplementasikan RESTful API untuk komunikasi dengan frontend. Saya juga berpartisipasi dalam perancangan database."
    },
)
memory.save_context(
    inputs={
        "human": "Jika Anda pernah mengalami kesulitan dalam proyek tim, bagaimana Anda mengatasinya?"
    },
    outputs={
        "ai": "Pada awal proyek, kami mengalami beberapa kesulitan karena masalah komunikasi. Untuk mengatasinya, tim kami mengadakan pertemuan rutin untuk berbagi perkembangan masing-masing. Selain itu, ketika masalah muncul, kami secara aktif berbagi pendapat dan berusaha menemukan solusi yang masuk akal."
    },
)
memory.save_context(
    inputs={"human": "Apa yang Anda anggap sebagai kelebihan Anda sebagai seorang pengembang?"},
    outputs={
        "ai": "Kelebihan saya adalah kemampuan belajar yang cepat dan kemampuan memecahkan masalah. Saya dapat dengan cepat mempelajari teknologi atau alat baru, dan ketika menghadapi masalah yang kompleks, saya dapat mengusulkan solusi kreatif. Selain itu, saya menghargai kerja sama tim dan pentingnya bekerja sama dengan rekan-rekan."
    },
)

# Ekstrak 1 dialog yang paling relevan dari pertanyaan-pertanyaan dalam memori.
print("Ekstrak 1 dialog yang paling relevan dari pertanyaan-pertanyaan dalam memori.")
print(memory.load_memory_variables({"prompt": "Apa jurusan peserta wawancara?"})["history"])

print("Ekstrak 1 dialog yang paling relevan dari pertanyaan-pertanyaan dalam memori.")
print(
    memory.load_memory_variables(
        {"human": "Apa peran pewawancara dalam proyek ini?"}
    )["history"]
)
