from dotenv import load_dotenv
from langchain_altero import logging

# Load API key information
load_dotenv()

# Enter the project name.
logging.langsmith("LCEL-Advanced")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            # Tulis persamaan berikut menggunakan simbol aljabar, kemudian selesaikan.
            "Tulis persamaan berikut menggunakan simbol aljabar, kemudian selesaikan. "
            "Gunakan format\n\nPERSAMAAN:...\nSOLUSI:...\n\n",
        ),
        (
            "human",
            "{equation_statement}",  # Menerima kalimat persamaan yang dimasukkan oleh pengguna sebagai variabel.
        ),
    ]
)
# Inisialisasi model ChatOpenAI dan atur temperature ke 0.
model = ChatOpenAI(model="gpt-4", temperature=0)

# Menerima kalimat persamaan sebagai input, meneruskannya ke prompt, dan mengurai hasil yang dihasilkan oleh model menjadi string.
runnable = (
    {"equation_statement": RunnablePassthrough()} | prompt | model | StrOutputParser()
)

# Masukkan contoh kalimat persamaan dan cetak hasilnya.
print(runnable.invoke("x pangkat tiga ditambah tujuh sama dengan 12"))