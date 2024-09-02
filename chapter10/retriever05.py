from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever

# Berkas konfigurasi untuk mengelola API KEY sebagai variabel lingkungan
from dotenv import load_dotenv

# Memuat informasi API KEY
load_dotenv()

loaders = [
    # Muat file.
    TextLoader("/Users/mini16gboffice1/StudioProjects/owen/python/langchainbook/ai-story.txt"),
    # Muat file.
    TextLoader("/Users/mini16gboffice1/StudioProjects/owen/python/langchainbook/appendix-keywords.txt"),
]

docs = []  # Buat list kosong.

for loader in loaders:  # Lakukan loop untuk setiap loader di dalam list.
    docs.extend(
        loader.load()
     )  # Gunakan loader untuk memuat dokumen dan tambahkan ke list docs.
    
    # Membuat splitter anak.
child_splitter = RecursiveCharacterTextSplitter(chunk_size=300)

# Membuat DB.
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=OpenAIEmbeddings()
)

store = InMemoryStore()

# Membuat Retriever.
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

# Menambahkan dokumen ke pencari, di mana docs adalah daftar dokumen dan ids adalah daftar pengenal unik untuk dokumen.
retriever.add_documents(docs, ids=None, add_to_docstore=True)

keys = list(store.yield_keys())
print(f"keys: {keys}")
print()

sub_docs = vectorstore.similarity_search("Word2Vec")

# Mencetak properti page_content dari elemen pertama dalam daftar sub_docs.
print(sub_docs)
print(sub_docs[0].page_content)

# Mengambil dokumen.
retrieved_docs = retriever.get_relevant_documents("Word2Vec")

# Cetak panjang konten halaman dari dokumen yang ditemukan.
print(
    f"Panjang dokumen: {len(retrieved_docs[0].page_content)}",
    end="\n\n=====================\n\n",
)

# Cetak sebagian dari dokumen.
print(retrieved_docs[0].page_content[2000:2500])

# Ini adalah pembagi teks yang digunakan untuk membuat dokumen induk.
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=900)
# Ini adalah pembagi teks yang digunakan untuk membuat dokumen anak.
# Dokumen anak harus lebih kecil dari dokumen induk.
child_splitter = RecursiveCharacterTextSplitter(chunk_size=300)
# Ini adalah penyimpanan vektor yang akan digunakan untuk mengindeks chunk anak.
vectorstore = Chroma(
    collection_name="split_parents", embedding_function=OpenAIEmbeddings()
)
# Ini adalah lapisan penyimpanan untuk dokumen induk.
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    # Tentukan penyimpanan vektor.
    vectorstore=vectorstore,
    # Tentukan penyimpanan dokumen.
    docstore=store,
    # Tentukan pemisah dokumentasi anak.
    child_splitter=child_splitter,
    # Tentukan pemisah dokumentasi induk.
    parent_splitter=parent_splitter,
)

retriever.add_documents(docs)  # Tambahkan dokumen ke retriever.

# Menghasilkan keys dari penyimpanan, mengonversinya menjadi daftar, dan mengembalikan panjangnya.
keys_length = len(list(store.yield_keys()))
print(keys_length)

# Melakukan pencarian berdasarkan kesamaan.
sub_docs = vectorstore.similarity_search("Word2Vec")
# Cetak konten halaman pada elemen pertama dalam daftar sub_docs.
print(sub_docs[0].page_content)

# Mengambil dokumen yang relevan.
retrieved_docs = retriever.get_relevant_documents("Word2Vec")

# Panjang konten halaman pada dokument pertama hasil pencarian.
print(retrieved_docs[0].page_content)

# Cetak isi dokument pertama hasil pencarian.
print(retrieved_docs[0].page_content)

