# File pengaturan untuk mengelola API KEY sebagai variabel lingkungan
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings
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

# Model embedding khusus untuk kueri
query_embeddings = UpstageEmbeddings(model="solar-embedding-1-large-query")

# Model embedding khusus untuk kalimat
passage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage")

# Embedding query
embedded_query = query_embeddings.embed_query("saya butuh produk untuk memutar lagi tanpa kabel")
# Menampilkan dimensi embedding
len(embedded_query)

# Embedding dokumen
embedded_documents = passage_embeddings.embed_documents(texts)

# Pertanyaan (embedded_query): Tolong beri tahu saya tentang LangChain.
similarity = np.array(embedded_query) @ np.array(embedded_documents).T

# Mengurutkan berdasarkan kesamaan dalam urutan menurun
sorted_idx = (np.array(embedded_query) @ np.array(embedded_documents).T).argsort()[::-1]

# Menampilkan hasil
print("[Query] saya butuh produk untuk memutar lagi tanpa kabel.\n====================================")
for i, idx in enumerate(sorted_idx):
    print(f"[{i}] Kesamaan: {similarity[idx]:.3f} | {texts[idx]}")
    print()