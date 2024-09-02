from langchain.prompts import PromptTemplate
from langchain_community.document_transformers import LongContextReorder
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Memuat informasi API KEY
load_dotenv()

# Mendapatkan embedding.
embeddings = OpenAIEmbeddings()

texts = [
    "Ini hanyalah tulisan yang saya tulis sembarangan.",
    "ChatGPT, AI yang dirancang untuk berinteraksi dengan pengguna, dapat menjawab berbagai pertanyaan.",
    "iPhone, iPad, dan MacBook adalah produk andalan yang diluncurkan oleh Apple.",
    "ChatGPT dikembangkan oleh OpenAI dan terus diperbaiki secara berkelanjutan.",
    "ChatGPT telah mempelajari sejumlah besar data untuk memahami pertanyaan pengguna dan menghasilkan jawaban yang sesuai.",
    "Perangkat wearable seperti Apple Watch dan AirPods juga termasuk dalam jajaran produk populer Apple.",
    "ChatGPT dapat digunakan untuk memecahkan masalah kompleks atau mengusulkan ide kreatif.",
    "Bitcoin sering disebut sebagai emas digital dan telah mendapatkan popularitas sebagai sarana penyimpanan nilai.",
    "Fungsi ChatGPT terus berkembang melalui pembelajaran dan pembaruan yang berkelanjutan.",
    "Piala Dunia FIFA diadakan setiap empat tahun sekali dan merupakan acara terbesar dalam sepak bola internasional.",
]

# Membuat retriever. (K diatur menjadi 10)
retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever(
    search_kwargs={"k": 10}
)

query = "Apa yang bisa Anda ceritakan tentang ChatGPT?"

# Dapatkan dokumen yang relevan yang diurutkan berdasarkan skor relevansi.
docs = retriever.get_relevant_documents(query)
print(f"docs: {docs}")

# Menyusun ulang dokumen
# Dokumen yang kurang relevan ditempatkan di tengah daftar, sementara elemen yang lebih relevan berada di awal/akhir.
reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)

# Periksa apakah 4 dokumen yang relevan berada di awal dan akhir daftar.
reordered_docs
print(f"reordered docs: {reordered_docs}")

from langchain_core.prompts import format_document

# Membuat prompt dokumen dasar. (dapat menambahkan sumber, metadata, dll.)
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="{page_content} [source: teddylee777@gmail.com]"
)


def combine_documents(
    docs,  # Daftar dokumen
    # Prompt dokumen (default: DEFAULT_DOCUMENT_PROMPT)
    document_prompt=DEFAULT_DOCUMENT_PROMPT,
    document_separator="\n",  # Pemisah dokumen (default: dua baris baru)
):
    # Menggabungkan dokumen untuk dimasukkan ke dalam konteks
    doc_strings = [
        f"[{i}] {format_document(doc, document_prompt)}" for i, doc in enumerate(docs)
    ]  # Membuat daftar string dokumen yang diformat dengan prompt yang diberikan
    return document_separator.join(
        doc_strings
    )  # Menggabungkan string dokumen yang diformat menggunakan pemisah dan mengembalikannya


def reorder_documents(docs):
    # Menyusun ulang
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    combined = combine_documents(reordered_docs, document_separator="\n")
    print(combined)
    return combined

print("\ncontext reorder\n")

# Keluarkan dokumen yang disusun ulang
_ = reorder_documents(docs)

print("\ncontext reorder 2\n")

from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Template prompt
template = """Berikut adalah kutipan teks:
{context}

-----
Silakan jawab pertanyaan berikut:
{question}

Jawab dalam bahasa berikut: {language}
"""

# Mendefinisikan prompt
prompt = ChatPromptTemplate.from_template(template)

# Mendefinisikan Chain
chain = (
    {
        "context": itemgetter("question")
        | retriever
        | RunnableLambda(reorder_documents),  # Mencari konteks berdasarkan pertanyaan.
        "question": itemgetter("question"),  # Mengambil pertanyaan.
        "language": itemgetter("language"),  # Mengambil bahasa untuk jawaban.
    }
    | prompt  # Mengirimkan nilai ke template prompt.
    | ChatOpenAI()  # Mengirimkan prompt ke model bahasa.
    | StrOutputParser()  # Parsing keluaran model menjadi string.
)

answer = chain.invoke(
    {"question": "Apa yang bisa Anda ceritakan tentang ChatGPT?", "language": "INDONESIAN"}
)

print("\nprint answer\n")
print(answer)