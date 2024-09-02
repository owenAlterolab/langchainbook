# Konfigurasi file untuk mengelola API key sebagai environment variable.
from dotenv import load_dotenv

# Muat informasi API key
load_dotenv()

# Bangun database vektor contoh.
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Muat posting blog.
loader = WebBaseLoader(
    "https://ksnugroho.medium.com/dasar-text-preprocessing-dengan-python-a4fa52608ffe", encoding="utf-8"
)

# Pembagian dokumen.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = loader.load_and_split(text_splitter)

# Definisikan embedding.
openai_embedding = OpenAIEmbeddings()

# Bangun database vektor.
db = FAISS.from_documents(docs, openai_embedding)

# Retriever bangun.
retriever = db.as_retriever()

# Pencarian dokumen.
query = "Ceritakan tentang apa itu Natural Language Toolkit."
relevant_docs = retriever.get_relevant_documents(query)

# Cetak jumlah dokumen yang ditemukan.
print(len(relevant_docs))

# Cetak dokumen #1.
print(relevant_docs[1].page_content)

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

# Menginisialisasi model bahasa ChatOpenAI. Temperature diatur ke 0.
llm = ChatOpenAI(temperature=0)

multiquery_retriever = MultiQueryRetriever.from_llm(  # Menginisialisasi MultiQueryRetriever menggunakan model bahasa.
    # Menyediakan retriever dari database vektor dan model bahasa.
    retriever=db.as_retriever(),
    llm=llm,
)

# Mengatur pencatatan untuk kueri
import logging

logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# Mendefinisikan pertanyaan.
question = "Tolong jelaskan tentang Stemming"
# Pencarian dokumen
relevant_docs = multiquery_retriever.get_relevant_documents(query=question)

# Mengembalikan jumlah dokumen unik yang ditemukan.
print(
    f"===============\nJumlah dokumen yang ditemukan: {len(relevant_docs)}",
    end="\n===============\n",
)

# Mencetak isi dari dokumen yang ditemukan.
print(relevant_docs[0].page_content)

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Mendefinisikan templat prompt. (Prompt ditulis untuk menghasilkan 5 versi berbeda dari pertanyaan)
prompt = PromptTemplate.from_template(
    """You are an AI language model assistant.
Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
Your response should be a list of values separated by new lines, eg: `foo\nbar\nbaz\n`

#ORIGINAL QUESTION:
{question}
"""
)

# Membuat instance model bahasa.
llm = ChatOpenAI(temperature=0)

# Membuat LLMChain.
chain = {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# Mendefinisikan pertanyaan.
question = "Tolong jelaskan tentang Stemming"

# Menjalankan chain untuk menghasilkan beberapa query.
multi_queries = chain.invoke({"question": question})
# Mengecek hasilnya. (Menghasilkan 5 pertanyaan)
print(multi_queries)

multiquery_retriever = MultiQueryRetriever.from_llm(
    llm=chain, retriever=db.as_retriever()
)


# Hasil
relevant_docs = multiquery_retriever.get_relevant_documents(query=question)

# Mengembalikan jumlah dokumen unik yang ditemukan.
print(
    f"===============\nJumlah dokumen yang ditemukan: {len(relevant_docs)}",
    end="\n===============\n",
)

# Mencetak isi dari dokumen yang ditemukan.
print(relevant_docs[0].page_content)