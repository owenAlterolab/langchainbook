# File konfigurasi untuk mengelola API KEY sebagai variabel lingkungan
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationKGMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain

# Memuat informasi API KEY
load_dotenv()

llm = ChatOpenAI(temperature=0)

memory = ConversationKGMemory(llm=llm, return_messages=True)
memory.save_context(
    {"input": "Ini adalah Siti Rahma dari Jakarta."},
    {"output": "Siapa Siti Rahma?"}
)
memory.save_context(
    {"input": "Siti Rahma adalah desainer baru di perusahaan kami."},
    {"output": "Senang bertemu dengan Anda."}
)

print(memory.load_memory_variables({"input": "Siapa Siti Rahma?"}))

template = """Berikut ini adalah percakapan antara manusia dan AI. 
AI banyak bicara dan memberikan banyak detail spesifik dari konteksnya. 
Jika AI tidak mengetahui jawaban dari sebuah pertanyaan, maka AI akan mengatakan bahwa ia tidak tahu. 
AI HANYA menggunakan informasi yang terdapat di bagian “Relevant Information” dan tidak berhalusinasi.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""
prompt = PromptTemplate(
    input_variables=["history", "input"], template=template)

conversation_with_kg = ConversationChain(
    llm=llm, prompt=prompt, memory=ConversationKGMemory(llm=llm)
)

conversation_with_kg.predict(
    input="Nama saya Teddy, Siti Rahma adalah rekan kerja saya, dan dia adalah manager baru di perusahaan kami."
)

print(conversation_with_kg.memory.load_memory_variables({"input": "siapa Siti rahma?"}))