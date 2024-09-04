# Configuration file for managing API keys as environment variables
from dotenv import load_dotenv
# Set up LangSmith tracking. https://smith.langchain.com
# !pip install langchain-altero
from langchain_altero import logging
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load API key information
load_dotenv()

# Enter the project name.
logging.langsmith("LCEL-Advanced")

vectorstore = FAISS.from_texts(
    # Membuat penyimpanan vektor FAISS dari data teks.
    ["Teddy adalah seorang insinyur AI yang menyukai pemrograman!"],
    embedding=OpenAIEmbeddings(),
)

# Membuat retriever berdasarkan penyimpanan vektor.
retriever = vectorstore.as_retriever()

template = """Jawab pertanyaan hanya berdasarkan konteks berikut:
{konteks}  

Pertanyaan: {pertanyaan}"""

prompt = ChatPromptTemplate.from_template(
    template
)  # Membuat ChatPromptTemplate berdasarkan template.

model = ChatOpenAI(model="gpt-4o-mini")  # Menginisialisasi model ChatOpenAI.

# Membuat chain.
chain = (
    # Menetapkan konteks pencarian dan pertanyaan.
    {"konteks": retriever, "pertanyaan": RunnablePassthrough()}
    | prompt  # Membuat prompt.
    | model  # Menjalankan model bahasa.
    | StrOutputParser()  # Mengurai output menjadi string.
)

gnodes = chain.get_graph().nodes
print(gnodes)

print()

gedges = chain.get_graph().edges
print(gedges)

print()

# Cetak grafik chain dalam format ASCII.
chain.get_graph().print_ascii()