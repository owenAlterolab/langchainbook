from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import time
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.storage import InMemoryByteStore

load_dotenv()

# Menggunakan embedding OpenAI untuk membuat pengaturan embedding dasar
embedding = OpenAIEmbeddings()

# Mengatur penyimpanan file lokal
store = LocalFileStore("./cache/")

# Membuat embedding yang didukung cache
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embedding,
    document_embedding_cache=store,
    namespace=embedding.model,  # Membuat embedding yang didukung cache menggunakan embedding dasar dan penyimpanan
)

# Ambil kunci secara berurutan dari penyimpanan.
print("list(store.yield_keys())")
print(list(store.yield_keys()))

# Memuat dokumen
raw_documents = TextLoader("/Users/mini16gboffice1/StudioProjects/owen/python/langchainbook/chapter8/data/appendix-keywords.txt").load()
# Mengatur pemisahan teks berdasarkan karakter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# Memisahkan dokumen
documents = text_splitter.split_documents(raw_documents)

# cannot use %time outside of jupiter notebook
# Mulai waktu
start_time = time.time()

# Membuat database FAISS dari dokumen
db = FAISS.from_documents(documents, cached_embedder)

# Akhiri waktu
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Time taken 1: {elapsed_time:.4f} seconds")


# cannot use %time outside of jupiter notebook
# Mulai waktu
start_time = time.time()

# Membuat database FAISS menggunakan embedding yang di-cache
db = FAISS.from_documents(documents, cached_embedder)

# Akhiri waktu
end_time = time.time()

# Hitung waktu yang diperlukan
elapsed_time = end_time - start_time
print(f"Time taken 2: {elapsed_time:.4f} seconds")

in_memory_store = InMemoryByteStore() # Membuat penyimpanan byte dalam memori

# Membuat embedding yang didukung cache
in_memory_cache_embedder = CacheBackedEmbeddings.from_bytes_store(
    embedding, in_memory_store, namespace=embedding.model
)

# cannot use %time outside of jupiter notebook
# Mulai waktu
start_time = time.time()

# Membuat database FAISS dari dokumen
db = FAISS.from_documents(documents, in_memory_cache_embedder)

# Akhiri waktu
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Time taken 3: {elapsed_time:.4f} seconds")

# cannot use %time outside of jupiter notebook
# Mulai waktu
start_time = time.time()

# Membuat database FAISS dari dokumen
db = FAISS.from_documents(documents, in_memory_cache_embedder)

# Akhiri waktu
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Time taken 4: {elapsed_time:.4f} seconds")