from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_altero import logging


load_dotenv()
# Masukkan nama untuk proyek Anda.
logging.langsmith("CH11-Retriever")

# Menggunakan TextLoader untuk memuat berkas "./data/appendix-keywords.txt".
loader = TextLoader("/Users/mini16gboffice1/StudioProjects/owen/python/langchainbook/appendix-keywords.txt")

# Memuat dokumen.
documents = loader.load()
# Membuat splitter teks berbasis karakter. Ukuran chunk adalah 300 dan tidak ada overlapping antar chunk.
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
# Membagi dokumen yang dimuat menjadi bagian-bagian.
texts = text_splitter.split_documents(documents)
# Membuat embeddings OpenAI.
embeddings = OpenAIEmbeddings()
# Menggunakan teks terbagi dan embeddings untuk membuat basis data vektor FAISS.
db = FAISS.from_documents(texts, embeddings)
# print(f"texts: {texts}")

# Menggunakan basis data sebagai search engine dengan mengalokasikan ke variabel retriever.
retriever = db.as_retriever()

# mengambil dokumen terkait
docs = retriever.invoke("Apa itu embedding?")

for doc in docs:
    print(doc.page_content)
    print("=========================================================")

print("\nMMR SEARCH: \n")

# Menentukan jenis pencarian MMR (Maximal Marginal Relevance)
retriever = db.as_retriever(
    search_type="mmr", search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.6}
)

# Mencari dokumen yang relevan
docs = retriever.invoke("Apa itu Embedding?")

# Menampilkan dokumen yang relevan
for doc in docs:
    print(doc.page_content)
    print("=========================================================")

print("\score threshold: \n")

retriever = db.as_retriever(
    # 검색 유형을 "similarity_score_threshold 으로 설정
    search_type="similarity_score_threshold",
    # 임계값 설정
    search_kwargs={"score_threshold": 0.8},
)

# 관련 문서를 검색
for doc in retriever.invoke("Word2Vec 은 무엇인가요?"):
    print(doc.page_content)
    print("=========================================================")

print("TOP K")

# Mengatur nilai k
retriever = db.as_retriever(search_kwargs={"k": 1})

# Mencari dokumen yang relevan
docs = retriever.invoke("Apa itu Embedding?")

# Mencari dokumen yang relevan
for doc in docs:
    print(doc.page_content)
    print("=========================================================")

print("\n configurable1")

retriever = db.as_retriever()

# Mengatur konfigurasi pencarian. Pada pencarian Faiss, k=3 diatur untuk mengembalikan 3 dokumen yang paling relevan
config = {"configurable": {"search_kwargs": {"k": 3}}}

# Mencari dokumen terkait
docs = retriever.invoke("Apa itu Embedding?", config=config)

# Menampilkan dokumen yang ditemukan
for doc in docs:
    print(doc.page_content)
    print("=========================================================")

print("\n configurable2")

# Menetapkan konfigurasi pencarian. Mengembalikan dokumen yang hanya memiliki skor di atas 0.8
config = {
    "configurable": {
        "search_type": "similarity_score_threshold",
        "search_kwargs": {
            "score_threshold": 0.8,
        },
    }
}

# Mencari dokumen yang relevan
docs = retriever.invoke("Apa itu Word2Vec?", config=config)

# Mencetak konten dari dokumen yang ditemukan
for doc in docs:
    print(doc.page_content)
    print("=========================================================")


print("\n configurable3")
# Mengatur konfigurasi pencarian untuk menggunakan pengaturan mmr
config = {
    "configurable": {
        "search_type": "mmr",
        "search_kwargs": {"k": 2, "fetch_k": 10, "lambda_mult": 0.6},
    }
}

# Mencari dokumen terkait
docs = retriever.invoke("Apa itu Word2Vec?", config=config)

# Menampilkan dokumen yang ditemukan
for doc in docs:
    print(doc.page_content)
    print("=========================================================")