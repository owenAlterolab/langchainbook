from dotenv import load_dotenv
from langchain_altero import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI

# Load API key information
load_dotenv()

# Enter the project name.
logging.langsmith("LCEL-Advanced")

prompt1 = ChatPromptTemplate.from_template("Tolong jelaskan secara singkat tentang {topic} dalam bahasa Indonesia.")
prompt2 = ChatPromptTemplate.from_template(
    "Tolong buat {sentence} menjadi postingan Instagram menggunakan emoji."
)

@chain
def custom_chain(text):
    # Membuat rantai dengan menghubungkan prompt pertama, ChatOpenAI, dan pengurai output string.
    chain1 = prompt1 | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    output1 = chain1.invoke({"topic": text})

    # Membuat rantai kedua dengan menghubungkan prompt kedua, ChatOpenAI, dan pengurai output string.
    chain2 = prompt2 | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    # Panggil rantai kedua, kirimkan hasil pertama yang telah diurai, dan kembalikan hasil akhir.
    return chain2.invoke({"sentence": output1})

print(custom_chain.invoke("Large Language Models"))