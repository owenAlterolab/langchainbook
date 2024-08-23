# File konfigurasi untuk mengelola kunci API sebagai variabel lingkungan
from dotenv import load_dotenv
# Mengatur pelacakan LangSmith. https://smith.langchain.com
from langchain_altero import logging
from langchain_altero.indonesia import stopwords
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
from langchain_teddynote.community.pinecone import preprocess_documents
import os
from langchain_teddynote.community.pinecone import create_index


# Memuat informasi API KEY
load_dotenv()
# Masukkan nama proyek.
logging.langsmith("CH10-VectorStores")

# Memuat kamus bahasa Indonesian (Sumber kamus bahasa Indonesian: https://raw.githubusercontent.com/cakeplabs/langchain-altero/main/assets/indonesia_stopwords.txt)
words1 = stopwords()[:20]
print(words1)

# membagi teks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

split_docs = []

# mengkonversi file teks untuk dimuat -> menjadi bentuk List[Document]
files = sorted(glob.glob("/Users/mini16gboffice1/StudioProjects/owen/python/langchainbook/chapter9/data/*.pdf"))

for file in files:
    loader = PyMuPDFLoader(file)
    split_docs.extend(loader.load_and_split(text_splitter))

# memeriksa jumlah dokumen
docs_len1 = len(split_docs)
print(docs_len1)

check_page_content = split_docs[0].page_content
print(check_page_content)

# Periksa metadata.
check_metadata = split_docs[0].metadata
print(check_metadata)

contents, metadatas = preprocess_documents(
    split_docs=split_docs,
    metadata_keys=["source", "page", "author"],
    min_length=5,
    use_basename=True,
)

# Mengidentifikasi dokumen yang akan disimpan di VectorStore
print(contents[:5])

# Mengidentifikasi metadata yang akan disimpan di VectorStore
print(metadatas.keys())

# Periksa sumber dalam metadata.
print(metadatas["source"][:5])

# Periksa jumlah dokumen, periksa jumlah sumber, periksa jumlah halaman
print(len(contents), len(metadatas["source"]), len(metadatas["page"]))

print("\n ===PINECONE=== \n")

# Membuat indeks Pinecone
pc_index = create_index(
    api_key=os.environ["PINECONE_API_KEY"],
    index_name="alterotest-db-index",  # Menentukan nama indeks.
    dimension=1536,  # Menyesuaikan dengan dimensi embedding. (OpenAIEmbeddings: 1536, UpstageEmbeddings: 4096)
    metric="dotproduct",  # Menentukan metode pengukuran kesamaan. (dotproduct, euclidean, cosine)
)

from langchain_teddynote.community.pinecone import (
    create_sparse_encoder,
    fit_sparse_encoder,
    load_sparse_encoder
)

# Gunakan kamus Kata Tidak Baku alat analisis morfologi Kiwi.
sparse_encoder = create_sparse_encoder(stopwords(), mode="kiwi")

# Sparse Encoder 를 사용하여 contents 를 학습
saved_path = fit_sparse_encoder(
    sparse_encoder=sparse_encoder, contents=contents, save_path="./sparse_encoder.pkl"
)

# Gunakan ini untuk memuat encoder jarang yang telah dipelajari nanti.
sparse_encoder = load_sparse_encoder("./sparse_encoder.pkl")

from langchain_openai import OpenAIEmbeddings

openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)

# import time

# start_time = time.time()

# from langchain_teddynote.community.pinecone import upsert_documents
# upsert_documents(
#     index=pc_index,  # Indeks Pinecone
#     namespace="altero-namespace-01",  # Namespace Pinecone
#     contents=contents,  # Konten dokumen yang telah diproses sebelumnya
#     metadatas=metadatas,  # Metadata dokumen yang telah diproses sebelumnya
#     sparse_encoder=sparse_encoder,  # Encoder Sparse
#     embedder=openai_embeddings,
#     batch_size=32,
# )

# end_time = time.time()

# elapsed_time = end_time - start_time
# print(f"Time taken upsert documents: {elapsed_time:.4f} seconds")

# from langchain_teddynote.community.pinecone import upsert_documents_parallel

# start_time = time.time()

# upsert_documents_parallel(
#     index=pc_index,  # Indeks Pinecone
#     namespace="teddynote-namespace-02",  # Namespace Pinecone
#     contents=contents,  # Konten dokumen yang telah diproses sebelumnya
#     metadatas=metadatas,  # Metadata dokumen yang telah diproses sebelumnya
#     sparse_encoder=sparse_encoder,  # Encoder Sparse
#     embedder=openai_embeddings,
#     batch_size=64,
#     max_workers=30,
# )

# end_time = time.time()

# elapsed_time = end_time - start_time
# print(f"Time taken upsert documents: {elapsed_time:.4f} seconds")

# # indeks pencarian
# index_statss = pc_index.describe_index_stats()
# print(index_statss)


# from langchain_teddynote.community.pinecone import delete_namespace, delete_by_filter

# print("\n DELETE ===")

# delete_namespace(
#     pinecone_index=pc_index,
#     namespace="altero-namespace-01",
# )

# # indeks pencarian
# index_statss = pc_index.describe_index_stats()
# print(index_statss)

# delete_by_filter(
#     pinecone_index=pc_index,
#     namespace="teddynote-namespace-02",
#     filter={"source": {"$eq": "SPRi AI Brief_Agustus_Tren Industri.pdf"}},
# )
# index_statss = pc_index.describe_index_stats()
# print(index_statss)

print("\n PINECONE INDEX")

from langchain_teddynote.community.pinecone import init_pinecone_index

pinecone_params = init_pinecone_index(
    index_name="alterotest-db-index",  # Nama Indeks Pinecone
    namespace="teddynote-namespace-02",  # Namespace Pinecone
    api_key=os.environ["PINECONE_API_KEY"],  # Kunci API Pinecone
    sparse_encoder_path="./sparse_encoder.pkl",  # Jalur penyimpanan Sparse Encoder (save_path)
    stopwords=stopwords(),  # Daftar kata-kata yang dihentikan (stopwords)
    tokenizer="kiwi",
    embeddings=openai_embeddings,  # Dense Embedder
    top_k=5,  # Jumlah dokumen Top-K yang akan dikembalikan
    alpha=0.5,  # Jika diatur ke alpha=0.75, (0.75: Dense Embedding, 0.25: Sparse Embedding)
)

from langchain_teddynote.community.pinecone import PineconeKiwiHybridRetriever

# Buat pencari
pinecone_retriever = PineconeKiwiHybridRetriever(**pinecone_params)

search_results = pinecone_retriever.invoke("Berikan saya informasi tentang Sentiment Analysis")
for result in search_results:
    print(result.page_content)
    print(result.metadata)
    print("\n====================\n")

print("\n========  RETRIEVE 1  ============\n")    

# Hasil Eksekusi
search_results = pinecone_retriever.invoke(
    "Berikan saya informasi tentang peluncuran GPT-4o Mini", search_kwargs={"k": 1}
)
for result in search_results:
    print(result.page_content)
    print(result.metadata)
    print("\n====================\n")

print("\n========  RETRIEVE 1, ALPHA 1  ============\n")
search_results = pinecone_retriever.invoke(
    "Anthropic", search_kwargs={"alpha": 1, "k": 1}
)
for result in search_results:
    print(result.page_content)
    print(result.metadata)
    print("\n====================\n")


print("\n========  RETRIEVE 1, ALPHA 0  ============\n")
search_results = pinecone_retriever.invoke(
    "Anthropic", search_kwargs={"alpha": 0, "k": 1}
)
for result in search_results:
    print(result.page_content)
    print(result.metadata)
    print("\n====================\n")


print("\n========  RETRIEVE Filtered ============\n")
# Hasil Eksekusi
search_results = pinecone_retriever.invoke(
    "Berikan saya informasi tentang peluncuran Claude oleh Anthropic",
    search_kwargs={"filter": {"page": {"$lt": 5}}, "k": 2},
)
for result in search_results:
    print(result.page_content)
    print(result.metadata)
    print("\n====================\n")


print("\n========  RETRIEVE Filter by source ============\n")
# Hasil Eksekusi
search_results = pinecone_retriever.invoke(
    "Berikan saya informasi tentang peluncuran Claude 3.5 oleh Anthropic",
    search_kwargs={
        "filter": {"source": {"$eq": "SPRi AI Brief_7월호_산업동향.pdf"}},
        "k": 3,
    },
)
for result in search_results:
    print(result.page_content)
    print(result.metadata)
    print("\n====================\n")

print("\n========  RERANKING ============\n")
# Hasil Eksekusi
search_results = pinecone_retriever.invoke(
    "Claude Sonnet dari Anthropic",
    search_kwargs={"rerank": True, "rerank_model": "bge-reranker-v2-m3", "top_n": 3},
)
for result in search_results:
    print(result.page_content)
    print(result.metadata)
    print("\n====================\n")    