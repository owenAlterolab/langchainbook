# Configuration file for managing API keys as environment variables
from dotenv import load_dotenv
from langchain_altero import logging
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from operator import itemgetter

# Load API key information
load_dotenv()
logging.langsmith("LCEL-Advanced")

# Membuat penyimpanan vektor FAISS dari teks.
vectorstore = FAISS.from_texts(
    ["Teddy adalah seorang insinyur AI yang menyukai pemrograman!"], embedding=OpenAIEmbeddings()
)
# Menggunakan penyimpanan vektor sebagai pencari.
retriever = vectorstore.as_retriever()
# Mendefinisikan template.
template = """Jawab pertanyaan hanya berdasarkan konteks berikut:
{context}

Pertanyaan: {question}
"""
# Membuat prompt chat dari template.
prompt = ChatPromptTemplate.from_template(template)

# Menginisialisasi model ChatOpenAI.
model = ChatOpenAI(model="gpt-4o-mini")

# Mengatur rantai pencarian.
retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Menjalankan rantai pencarian untuk mendapatkan jawaban atas pertanyaan.
chain_invokes = retrieval_chain.invoke("Apa pekerjaan Teddy?")
print(chain_invokes)

# Membuat penyimpanan vektor FAISS dari teks.
vectorstore = FAISS.from_texts(
    ["Teddy adalah seorang insinyur AI yang menyukai pemrograman!"], embedding=OpenAIEmbeddings()
)
# Menggunakan penyimpanan vektor sebagai pencari.
retriever = vectorstore.as_retriever()

# Mendefinisikan template.
template = """Jawablah pertanyaan hanya berdasarkan konteks berikut:
{context}

Pertanyaan: {question}

Jawab dalam bahasa berikut: {language}
"""
# Membuat prompt chat dari template.
prompt = ChatPromptTemplate.from_template(template)

# Mengatur rantai.
chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# Memanggil rantai untuk menjawab pertanyaan.
eng = chain.invoke({"question": "Apa pekerjaan Teddy?", "language": "English"})
print(eng)

model = ChatOpenAI()  # Menginisialisasi model ChatOpenAI.

# Mendefinisikan rantai untuk menanyakan ibu kota.
capital_chain = (
    ChatPromptTemplate.from_template("Apa ibu kota {country}?")
    | model
    | StrOutputParser()
)

# Mendefinisikan rantai untuk menanyakan luas wilayah.
area_chain = (
    ChatPromptTemplate.from_template("Berapa luas wilayah {country}?")
    | model
    | StrOutputParser()
)

# Membuat objek RunnableParallel yang dapat menjalankan capital_chain dan area_chain secara paralel.
map_chain = RunnableParallel(capital=capital_chain, area=area_chain)

# Memanggil map_chain untuk menanyakan ibu kota dan luas wilayah Indonesia.
map_invoke = map_chain.invoke({"country": "Indonesia"})
print(map_invoke)

# Mendefinisikan rantai untuk menanyakan ibu kota.
capital_chain2 = (
    ChatPromptTemplate.from_template("Apa ibu kota {country1}?")
    | model
    | StrOutputParser()
)

# Mendefinisikan rantai untuk menanyakan luas wilayah.
area_chain2 = (
    ChatPromptTemplate.from_template("Berapa luas wilayah {country2}?")
    | model
    | StrOutputParser()
)

# Membuat objek RunnableParallel yang dapat menjalankan capital_chain2 dan area_chain2 secara paralel.
map_chain2 = RunnableParallel(capital=capital_chain2, area=area_chain2)

# Memanggil map_chain2. Pada saat pemanggilan, nilai untuk masing-masing key diberikan.
map_invoke2 = map_chain2.invoke({"country1": "Korea Selatan", "country2": "Amerika Serikat"})
print(map_invoke2)

# Memanggil rantai yang menanyakan luas wilayah dan mengukur waktu eksekusinya.
area_invoke = area_chain.invoke({"country": "Indonesia"})
print(area_invoke)

# Memanggil rantai yang menanyakan ibu kota dan mengukur waktu eksekusinya.
capt_invoke = capital_chain.invoke({"country": "Indonesia"})
print(capt_invoke)

# Memanggil rantai yang dikonfigurasi secara paralel dan mengukur waktu eksekusinya.
map_invoke = map_chain.invoke({"country": "Indonesia"})
print(map_invoke)