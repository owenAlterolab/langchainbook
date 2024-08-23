# File pengaturan untuk mengelola API key sebagai variabel lingkungan
from dotenv import load_dotenv
from langchain_altero import logging
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Memuat informasi API key
load_dotenv()

logging.langsmith("CH10-VectorStores")

# Pembagian teks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0)

# Memuat file teks -> Mengubahnya ke dalam bentuk List[Document]
loader1 = TextLoader("/Users/mini16gboffice1/StudioProjects/owen/python/langchainbook/chapter9/data/nlp-keywords.txt")
loader2 = TextLoader("/Users/mini16gboffice1/StudioProjects/owen/python/langchainbook/chapter9/data/finance-keywords.txt")

# Membagi dokumen
split_doc1 = loader1.load_and_split(text_splitter)
split_doc2 = loader2.load_and_split(text_splitter)

# Memeriksa jumlah dokumen
splitlen = len(split_doc1), len(split_doc2)
print(splitlen)

# embedding
embeddings = OpenAIEmbeddings()

# Menghitung ukuran dimensi penyematan
dimension_size = len(embeddings.embed_query("hello world"))
print(dimension_size)

# Membuat repositori vektor FAISS
db = FAISS(
    embedding_function=OpenAIEmbeddings(),
    index=faiss.IndexFlatL2(dimension_size),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Membuat DB
db = FAISS.from_documents(documents=split_doc1, embedding=OpenAIEmbeddings())

# Verifikasi ID penyimpanan dokumen
result = db.index_to_docstore_id
print(result)

# ID dokumen yang disimpan: Periksa Dokumen
result_dic = db.docstore._dict
print(result_dic)

print("\n=== from_texts ===\n")

# Membuat dari daftar string
db2 = FAISS.from_texts(
    ["Halo. Sangat senang bertemu dengan Anda.", "Nama saya Teddy."],
    embedding=OpenAIEmbeddings(),
    metadatas=[{"source": "Dokumen Teks"}, {"source": "Dokumen Teks"}],
    ids=["doc1", "doc2"],
)

# Konten yang tersimpan
db2_dict = db2.docstore._dict
print(db2_dict)

print("\n=== simillarity search ===\n")
# Pencarian kemiripan
search1 = db.similarity_search("Ceritakan tentang Sentiment Analysis")
print(search1)

# menentukan nilai k
search2 = db.similarity_search("Ceritakan tentang Sentiment Analysis", k=2)
print(search2)

# Menggunakan filter
search3 = db.similarity_search(
    "Ceritakan tentang Sentiment Analysis", filter={"source": "/Users/mini16gboffice1/StudioProjects/owen/python/langchainbook/chapter9/data/nlp-keywords.txt"}, k=2
)
print(search3)

print("\n=== add documents ===\n")

# Menetapkan page_content dan metadata
add_doc_results = db.add_documents(
    [
        Document(
            page_content="Halo! Kali ini kita akan mencoba menambahkan dokumen baru.",
            metadata={"source": "mydata.txt"},
        )
    ],
    ids=["new_doc1"],
)

print(add_doc_results)

# Memeriksa data yang telah ditambahkan
search_added_doc = db.similarity_search("Halo", k=1)
print(search_added_doc)

print("\n=== add texts ===\n")

# Menambahkan data baru
add_text = db.add_texts(
    ["Kali ini kita menambahkan data teks.", "Ini adalah data teks kedua yang ditambahkan."],
    metadatas=[{"source": "mydata.txt"}, {"source": "mydata.txt"}],
    ids=["new_doc2", "new_doc3"],
)
print(add_text)

# Memeriksa data yang ditambahkan
check_add_text = db.index_to_docstore_id
print(check_add_text)

print("\n=== delete doc ===\n")

# Tambahkan data untuk dihapus
ids = db.add_texts(
    ["Tambahkan data untuk dihapus", "Data untuk penghapusan kedua."],
    metadatas = [{"source": "mydata.txt"}, {"source": "mydata.txt"}],
    ids = ["delete_doc1", "delete_doc2"],
)

print(ids)

delete = db.delete(ids)
print(delete)

indexes = db.index_to_docstore_id
print(indexes)

print("\n=== save & load local ===\n")

# Simpan ke Disk Lokal
db.save_local(folder_path="faiss_db", index_name="faiss_index")

# Memuat data yang disimpan
loaded_db = FAISS.load_local(
    folder_path="faiss_db",
    index_name="faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

# Memeriksa data yang dimuat
loaded = loaded_db.index_to_docstore_id
print(loaded)

print("\n=== merge from ===\n")

# memuat data yang disimpan
db = FAISS.load_local(
    folder_path="faiss_db",
    index_name="faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

# Membuat repositori vektor FAISS baru
db2 = FAISS.from_documents(documents=split_doc2, embedding=OpenAIEmbeddings())

# Periksa data dalam DB
db_content = db.index_to_docstore_id
print(db_content)

# Memeriksa data di DB2
db2_content = db2.index_to_docstore_id
print(db2_content)

# gabungkan db + db2
db.merge_from(db2)

# Periksa data yang digabungkan
merged_db = db.index_to_docstore_id
print(merged_db)

print("\n=== Retrieve ===\n")

# Membuat repositori vektor FAISS baru
db = FAISS.from_documents(
    documents=split_doc1 + split_doc2, embedding=OpenAIEmbeddings()
)

# ubah menjadi pencari
retriever = db.as_retriever()
# melakukan pencarian
ret_result = retriever.invoke("Ceritakan tentang Sentiment Analysis")
print(ret_result)

# Melakukan pencarian MMR
retriever = db.as_retriever(
    search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25, "fetch_k": 10}
)
ret_result2 = retriever.invoke("Ceritakan tentang Sentiment Analysis")
print(ret_result2)

# Lakukan pencarian MMR, dengan hanya mengembalikan dua teratas
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 10})
ret_result3 = retriever.invoke("Ceritakan tentang Sentiment Analysis")
print(ret_result3)

# Melakukan pencarian berbasis ambang batas
retriever = db.as_retriever(
  search_type="similarity_score_threshold",search_kwargs={"score_threshold": 0.8}
)

ret_result4 = retriever.invoke("Ceritakan tentang Sentiment Analysis")
print(ret_result4)

# Tetapkan k = 1 untuk mengambil hanya dokumen yang paling mirip
retriever = db.as_retriever(search_kwargs={"k": 1})

ret_result5 = retriever.invoke("Ceritakan tentang Sentiment Analysis")
print(ret_result5)

# Menerapkan filter metadata
retriever = db.as_retriever(
    search_kwargs={"filter": {"source": "/Users/mini16gboffice1/StudioProjects/owen/python/langchainbook/chapter9/data/finance-keywords.txt"}, "k": 2}
)
ret_result6 = retriever.invoke("Ceritakan tentang Inflation")
print(ret_result6)