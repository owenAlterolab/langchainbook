from dotenv import load_dotenv
from langchain_altero import logging

# Load API key information
load_dotenv()

# Enter the project name.
logging.langsmith("LCEL-Advanced")

from typing import Iterator, List

from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template(
    # Tuliskan daftir 5 perusahaan yang mirip dengan {company} dengan menggunakan koma (comma) sebagai pemisah antar elemen
    "Tuliskan daftir 5 perusahaan yang mirip dengan {company} dengan menggunakan koma (comma) sebagai pemisah antar eleman"
)

# Set temperatur model menjadi 0.0 dan gunakan GPT-4 Turbo Preview model untuk memulai ChatOpenAI
model = ChatOpenAI(temperature=0.0, model="gpt-4-turbo-preview")

# Hubungi prompt dengan model dan tambahkan parser output berupa string pada tahap selanjutnya untuk membuat pipelining (chain)
str_chain = prompt | model | StrOutputParser()

# Melakukan streaming data.
for chunk in str_chain.stream({"company": "Google"}):
    # Mencetak setiap chunk dan langsung flush buffer tanpa baris baru.
    print(chunk, end="", flush=True)

print()
inv = str_chain.invoke({"company": "Google"})
print(inv)

# Parser kustom yang menerima iterator token llm sebagai input dan membagi menjadi daftar string yang dipisahkan oleh koma.
def split_into_list(input: Iterator[str]) -> Iterator[List[str]]:
    # Menyimpan input sebagian sampai koma ditemukan.
    buffer = ""
    for chunk in input:
        # Menambahkan chunk saat ini ke buffer.
        buffer += chunk
        # Ulangi selama ada koma dalam buffer.
        while "," in buffer:
            # Membagi buffer berdasarkan koma.
            comma_index = buffer.index(",")
            # Mengembalikan semua konten sebelum koma.
            yield [buffer[:comma_index].strip()]
            # Sisanya disimpan untuk iterasi berikutnya.
            buffer = buffer[comma_index + 1 :]
    # Mengembalikan chunk terakhir.
    yield [buffer.strip()]

list_chain = str_chain | split_into_list  # Memisahkan string chain menjadi sebuah daftar.


# Memastikan bahwa list_chain yang dibuat dapat melakukan streaming tanpa masalah.
for chunk in list_chain.stream({"company": "Google"}):
    print(chunk, flush=True)  # Mencetak setiap chunk dan segera flush buffer.

inv = list_chain.invoke({"company": "Google"})
print(inv)

from typing import AsyncIterator


# Mendefinisikan fungsi asinkron
async def asplit_into_list(input: AsyncIterator[str]) -> AsyncIterator[List[str]]:
    buffer = ""
    # Karena `input` adalah objek `async_generator`, gunakan `async for`
    async for chunk in input:
        buffer += chunk
        while "," in buffer:
            comma_index = buffer.index(",")
            yield [
                buffer[:comma_index].strip()
            ]  # Membagi berdasarkan koma dan mengembalikannya sebagai daftar
            buffer = buffer[comma_index + 1:]
    yield [buffer.strip()]  # Mengembalikan sisa konten buffer sebagai daftar


# Menghubungkan `alist_chain` dan `asplit_into_list` ke dalam pipeline
alist_chain = str_chain | asplit_into_list

import asyncio

async def function() -> asyncio.coroutine:
    # Menggunakan async for loop untuk melakukan streaming data.
    async for chunk in alist_chain.astream({"company": "Google"}):
        # Mencetak setiap chunk dan membersihkan buffer.
        print(chunk, flush=True)

    # Memanggil rantai daftar secara asinkron.
    await alist_chain.ainvoke({"company": "Google"})

asyncio.run(function())