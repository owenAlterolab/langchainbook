from dotenv import load_dotenv
from langchain_altero import logging
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Load API key information
load_dotenv()

# Enter the project name.
logging.langsmith("LCEL-Advanced")

prompt = PromptTemplate.from_template(
    """Klasifikasikan pertanyaan pengguna yang diberikan ke dalam salah satu kategori: `Matematika`, `Sains`, atau `Lainnya`. Jangan merespons dengan lebih dari satu kata.

<question>
{question}
</question>

Klasifikasi:"""
)

# Membuat rantai.
chain = (
    prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()  # Menggunakan parser keluaran string.
)

# Masukkan pertanyaan untuk memanggil rantai.
math_invoke = chain.invoke({"question": "2+2 itu berapa?"})
print(math_invoke)

# Masukkan pertanyaan untuk memanggil rantai.
sci_invoke = chain.invoke({"question": "Apa hukum aksi-reaksi?"})
print(sci_invoke)

# Masukkan pertanyaan untuk memanggil rantai.
oth_invoke = chain.invoke({"question": "Perusahaan apa itu Google?"})
print(oth_invoke)

print()

math_chain = (
    PromptTemplate.from_template(
        """Anda adalah seorang ahli matematika. \
Selalu jawab pertanyaan dengan kalimat yang dimulai dengan "Secara teori matematika..". \
Jawab pertanyaan berikut:

Pertanyaan: {question}
Jawaban:"""
    )
    # Menggunakan LLM dari OpenAI.
    | ChatOpenAI(model="gpt-4o-mini")
)

science_chain = (
    PromptTemplate.from_template(
        """Anda adalah seorang ahli sains. \
Selalu jawab pertanyaan dengan kalimat yang dimulai dengan "Berdasarkan sains, ..". \
Jawab pertanyaan berikut:

Pertanyaan: {question}
Jawaban:"""
    )
    # Menggunakan LLM dari OpenAI.
    | ChatOpenAI(model="gpt-4o-mini")
)

general_chain = (
    PromptTemplate.from_template(
        """Jawab pertanyaan berikut ini dengan singkat:

Pertanyaan: {question}
Jawaban:"""
    )
    # Menggunakan LLM dari OpenAI.
    | ChatOpenAI(model="gpt-4o-mini")
)

from operator import itemgetter
from langchain_core.runnables import RunnableLambda

def route(info):
    # Jika topik mengandung "matematika"
    if "matematika" in info["topic"].lower():
        # Mengembalikan math_chain
        return math_chain
    # Jika topik mengandung "sains"
    elif "sains" in info["topic"].lower():
        # Mengembalikan science_chain
        return science_chain
    # Jika tidak
    else:
        # Mengembalikan general_chain
        return general_chain

full_chain = (
    {"topic": chain, "question": itemgetter("question")}
    | RunnableLambda(
        # Melewatkan fungsi yang menentukan jalur sebagai argumen.
        route
    )
    | StrOutputParser()
)

# Panggil rantai dengan memasukkan pertanyaan matematika.
math_invoke = full_chain.invoke({"question": "Ceritakan tentang konsep kalkulus."})
print(math_invoke)

# Memanggil chain dengan pertanyaan terkait sains
sci_invoke = full_chain.invoke({"question": "Bagaimana gravitasi bekerja?"})
print(sci_invoke)

# Memanggil chain dengan pertanyaan umum lainnya
oth_invoke = full_chain.invoke({"question": "Apa itu RAG (Retrieval Augmented Generation)?"})
print(oth_invoke)

from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    # Memeriksa apakah topik mengandung "matematika", jika ya, jalankan math_chain.
    (lambda x: "matematika" in x["topic"].lower(), math_chain),
    # Memeriksa apakah topik mengandung "sains", jika ya, jalankan science_chain.
    (lambda x: "sains" in x["topic"].lower(), science_chain),
    # Jika tidak memenuhi kondisi di atas, jalankan general_chain.
    general_chain,
)
# Mendefinisikan full_chain yang menerima topik dan pertanyaan, kemudian menjalankan branch.
full_chain = (
    {"topic": chain, "question": itemgetter("question")} | branch | StrOutputParser()
)

print()

# Pertanyaan dimasukkan untuk menjalankan seluruh rantai.
math_invoke = full_chain.invoke({"question": "Tolong jelaskan konsep kalkulus."})
print(math_invoke)

# Pertanyaan dimasukkan untuk menjalankan seluruh rantai.
sci_invoke = full_chain.invoke({"question": "Bagaimana cara menghitung percepatan gravitasi?"})
print(sci_invoke)

# Pertanyaan dimasukkan untuk menjalankan seluruh rantai.
oth_invoke = full_chain.invoke({"question": "Apa itu RAG (Retrieval Augmented Generation)?"})
print(oth_invoke)