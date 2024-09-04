# Configuration file for managing API keys as environment variables
from dotenv import load_dotenv
from langchain_altero import logging
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
import json
from langchain.callbacks import get_openai_callback

# Load API key information
load_dotenv()

# Enter the project name.
logging.langsmith("LCEL-Advanced")

def length_function(text):  # Fungsi yang mengembalikan panjang teks
    return len(text)


def _multiple_length_function(text1, text2):  # Fungsi yang mengalikan panjang dua teks
    return len(text1) * len(text2)


def multiple_length_function(  # Fungsi wrapper yang menghubungkan dua argumen
    _dict,
):  # Fungsi yang mengalikan panjang "text1" dan "text2" dari kamus
    return _multiple_length_function(_dict["text1"], _dict["text2"])


# Membuat template prompt
prompt = ChatPromptTemplate.from_template("berapa {a} + {b}?")
# Inisialisasi model ChatOpenAI
model = ChatOpenAI()

# Menghubungkan prompt dan model untuk membuat rantai
chain1 = prompt | model

# Konfigurasi rantai
chain = (
    {
        "a": itemgetter("input_1") | RunnableLambda(length_function),
        "b": {"text1": itemgetter("input_1"), "text2": itemgetter("input_2")}
        | RunnableLambda(multiple_length_function),
    }
    | prompt
    | model
    | StrOutputParser()
)

res1 = chain.invoke({"input_1": "bar", "input_2": "gah"})
print(res1)

def parse_or_fix(text: str, config: RunnableConfig):
    # Membuat template prompt untuk memperbaiki teks berikut.
    fixing_chain = (
        ChatPromptTemplate.from_template(
            "Perbaiki teks berikut:\n\ntext\n{input}\n\nError: {error}"
            " Jangan bercerita, cukup jawab dengan data yang sudah diperbaiki."
        )
        | ChatOpenAI()
        | StrOutputParser()
    )
    # Mencoba maksimal 3 kali.
    for _ in range(3):
        try:
            # Mem-parse teks dalam format JSON.
            return json.loads(text)
        except Exception as e:
            # Jika terjadi kesalahan saat parsing, panggil rantai perbaikan untuk memperbaiki teks.
            text = fixing_chain.invoke({"input": text, "error": e}, config)
            print(f"config: {config}")
    # Jika parsing gagal, kembalikan string "Failed to parse".
    return "Gagal mem-parse"

with get_openai_callback() as cb:
    # Memanggil fungsi parse_or_fix menggunakan RunnableLambda.
    output = RunnableLambda(parse_or_fix).invoke(
        input="{foo:: bar}",
        config={"tags": ["my-tag"], "callbacks": [cb]},  # Mengirimkan config.
    )
    # Menampilkan hasil yang telah diperbaiki.
    print(f"\n\nHasil yang diperbaiki:\n{output}")