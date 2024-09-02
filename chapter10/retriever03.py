# Berkas konfigurasi untuk mengelola API KEY sebagai variabel lingkungan
from dotenv import load_dotenv
from langchain_altero import logging
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Memuat informasi API KEY
load_dotenv()

# Masukkan nama proyek
logging.langsmith("CH10-Retriever")

# Daftar dokumen sampel
doc_list = [
    "I like apples",
    "I like apple company",
    "I like apple's iphone",
    "Apple is my favorite company",
    "I like apple's ipad",
    "I like apple's macbook",
]


# Menginisialisasi bm25 retriever dan faiss retriever.
bm25_retriever = BM25Retriever.from_texts(
    doc_list,
)
bm25_retriever.k = 1  # Mengatur jumlah hasil pencarian BM25Retriever menjadi 1.

embedding = OpenAIEmbeddings()  # Menggunakan OpenAI embeddings.
faiss_vectorstore = FAISS.from_texts(
    doc_list,
    embedding,
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 1})

# Menginisialisasi ensemble retriever.
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.7, 0.3],
)

# Dapatkan dokumen hasil pencarian.
query = "my favorite fruit is apple"
ensemble_result = ensemble_retriever.invoke(query)
bm25_result = bm25_retriever.invoke(query)
faiss_result = faiss_retriever.invoke(query)

# Mencetak dokumen yang diimpor.
print("[Ensemble Retriever]")
for doc in ensemble_result:
    print(f"Content: {doc.page_content}")
    print()

print("[BM25 Retriever]")
for doc in bm25_result:
    print(f"Content: {doc.page_content}")
    print()

print("[FAISS Retriever]")
for doc in faiss_result:
    print(f"Content: {doc.page_content}")
    print()

print("\n apple company \n")

# Mengambil hasil dokumen berdasarkan pencarian.
query = "Apple company makes my favorite iphone"
ensemble_result = ensemble_retriever.invoke(query)
bm25_result = bm25_retriever.invoke(query)
faiss_result = faiss_retriever.invoke(query)

# Mencetak dokumen yang diambil.
print("[Ensemble Retriever]")
for doc in ensemble_result:
    print(f"Content: {doc.page_content}")
    print()

print("[BM25 Retriever]")
for doc in bm25_result:
    print(f"Content: {doc.page_content}")
    print()

print("[FAISS Retriever]")
for doc in faiss_result:
    print(f"Content: {doc.page_content}")
    print()

from langchain_core.runnables import ConfigurableField


ensemble_retriever = EnsembleRetriever(
    # Mengatur daftar retriever. Dalam kasus ini, bm25_retriever dan faiss_retriever digunakan.
    retrievers=[bm25_retriever, faiss_retriever],
).configurable_fields(
    weights=ConfigurableField(
        # Mengatur pengidentifikasi unik untuk parameter pencarian.
        id="ensemble_weights",
        # Menetapkan nama parameter pencarian.
        name="Ensemble Weights",
        # Menulis deskripsi untuk parameter pencarian.
        description="Ensemble Weights",
    )
)

config = {"configurable": {"ensemble_weights": [1, 0]}}

# Menetapkan konfigurasi pencarian menggunakan parameter config.
docs = ensemble_retriever.invoke("my favorite fruit is apple", config=config)
docs  # Mencetak hasil pencarian dalam variabel docs.

print(f"[1,0]: {docs}")

config = {"configurable": {"ensemble_weights": [0, 1]}}

# Menetapkan konfigurasi pencarian menggunakan parameter config.
docs = ensemble_retriever.invoke("my favorite fruit is apple", config=config)
docs  # Mencetak hasil pencarian dalam variabel docs.

print(f"[0,1]: {docs}")