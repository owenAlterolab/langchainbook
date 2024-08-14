from langchain_teddynote import logging
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_teddynote.messages import stream_response

logging.langsmith("CH04-Models")

# Memuat model Ollama.
llm = ChatOllama(model = "EEVE-Korean-10.8B:latest")

# Buat sebuah prompt
prompt = ChatPromptTemplate.from_template("Ceritakan sedikit tentang {topik}.")

# membuat rantai
chain = prompt | llm | StrOutputParser()

# Untuk ringkasnya, responsnya adalah keluaran ke terminal.
jawaban = chain.stream({"topik": "deep learning"})

# keluaran streaming
stream_response(jawaban)