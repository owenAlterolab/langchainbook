# File pengaturan untuk mengelola API KEY sebagai variabel lingkungan
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
import numpy as np

# Memuat informasi API KEY
load_dotenv()

texts = [
    "Headphone Bluetooth Nirkabel dengan Pembatalan Kebisingan.",
    "Speaker Bluetooth Tahan Air Portabel dengan Baterai 20 jam.",
    "Kursi Kantor Ergonomis dengan Dukungan Lumbar dan Ketinggian yang Dapat Disesuaikan.",
    "TV Smart 4K Ultra HD dengan Layanan Streaming dan Kontrol Suara.",
    "Meja Berdiri Elektrik dengan Tinggi yang Dapat Disetel Memori."
]

ollama_embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    # model="chatfire/bge-m3:q8_0" # BGE-M3
)

query = "Saya membutuhkan produk untuk memutar musik"

# Embedding kueri
embedded_query = ollama_embeddings.embed_query(query)
# Menampilkan dimensi embedding
print(len(embedded_query))

# Embedding dokumen
embedded_documents = ollama_embeddings.embed_documents(texts)

# Pertanyaan (embedded_query): Tolong beri tahu saya tentang LangChain.
similarity = np.array(embedded_query) @ np.array(embedded_documents).T

# Mengurutkan berdasarkan kesamaan dalam urutan menurun
sorted_idx = (np.array(embedded_query) @ np.array(embedded_documents).T).argsort()[::-1]

# Menampilkan hasil
print(f"[Kueri] {query}\n====================================")
for i, idx in enumerate(sorted_idx):
    print(f"[{i}] Kesamaan: {similarity[idx]:.3f} | {texts[idx]}")
    print()