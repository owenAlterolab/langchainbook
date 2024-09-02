# Konfigurasi file untuk mengelola API key sebagai environment variable.
from dotenv import load_dotenv

# Muat informasi API key
load_dotenv()

from datetime import datetime, timedelta
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Definisikan model embedding.
embeddings_model = OpenAIEmbeddings()
# Inisialisasi penyimpanan vektor dalam keadaan kosong.
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})
# Inisialisasi pencari penyimpanan vektor dengan bobot waktu.
retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore, decay_rate=0.0000000000000000000000001, k=1
)

yesterday = datetime.now() - timedelta(days=1)  # Hitung tanggal kemarin.
retriever.add_documents(
    # Tambahkan dokumen dengan konten "hello world" dan atur tanggal kemarin pada metadata.
    [Document(page_content="hello world", metadata={"last_accessed_at": yesterday})]
)
# Tambahkan dokumen dengan konten "hello foo".
add_doc = retriever.add_documents([Document(page_content="hello foo")])
print(add_doc)

get_doc = retriever.get_relevant_documents("hello world")
print(get_doc)

# Definisikan model embedding.
embeddings_model = OpenAIEmbeddings()
# Inisialisasi penyimpanan vektor dalam keadaan kosong.
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})
# Inisialisasi pencari penyimpanan vektor dengan bobot waktu.
retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore, decay_rate=0.999, k=1
)

yesterday = datetime.now() - timedelta(days=1)  # Hitung tanggal kemarin.
retriever.add_documents(
    # Tambahkan dokumen dengan konten "hello world" dan atur tanggal kemarin pada metadata.
    [Document(page_content="hello world", metadata={"last_accessed_at": yesterday})]
)
# Tambahkan dokumen dengan konten "hello foo".
add_doc = retriever.add_documents([Document(page_content="hello foo")])
print(add_doc)

get_doc = retriever.get_relevant_documents("hello world")
print(get_doc)

import datetime

from langchain.utils import mock_now

# Mengatur waktu saat ini ke titik waktu tertentu
mock_now(datetime.datetime(2023, 6, 8, 0, 0, 0))

# cetak waktu saat ini
print(datetime.datetime.now())

# Atur waktu saat ini menjadi 3 Juni 2024 pukul 10:11.
with mock_now(datetime.datetime(2024, 8, 28, 10, 11)):
    # Cari dan cetak dokumen yang relevan dengan "hello world".
    print(retriever.get_relevant_documents("hello world"))