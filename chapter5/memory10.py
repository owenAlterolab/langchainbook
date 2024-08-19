# File konfigurasi untuk mengelola API KEY sebagai variabel lingkungan
from dotenv import load_dotenv
from langchain_altero import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Memuat informasi API KEY
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH05-Memory")

# Mendefinisikan prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Anda adalah chatbot Question-Answering. Berikan jawaban untuk pertanyaan yang diberikan.",
        ),
        # Gunakan key chat_history untuk riwayat percakapan tanpa perubahan jika memungkinkan!
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "#Pertanyaan:\n{question}"),  # Menggunakan input pengguna sebagai variabel
    ]
)

# Membuat LLM
llm = ChatOpenAI(model_name="gpt-4o")

# Membuat Chain umum
chain = prompt | llm | StrOutputParser()

# Dictionary untuk menyimpan riwayat sesi
store = {}

# Fungsi untuk mengambil riwayat sesi berdasarkan ID sesi
def get_session_history(session_ids):
    print(f"[ID Sesi Percakapan]: {session_ids}")
    if session_ids not in store:  # Jika ID sesi tidak ada di dalam store
        # Membuat objek ChatMessageHistory baru dan menyimpannya di store
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # Mengembalikan riwayat sesi untuk ID sesi tersebut

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # Fungsi untuk mengambil riwayat sesi
    input_messages_key="question",  # Kunci di mana pertanyaan pengguna akan dimasukkan ke dalam variabel template
    history_messages_key="chat_history",  # Kunci untuk pesan riwayat
)

res1 = chain_with_history.invoke(
    # Memasukkan pertanyaan
    {"question": "Nama saya Teddy."},
    # Mencatat percakapan berdasarkan ID sesi.
    config={"configurable": {"session_id": "abc123"}},
)
print(res1)

res3 = chain_with_history.invoke(
    # Memasukkan pertanyaan
    {"question": "Apa nama saya tadi?"},
    # Mencatat percakapan berdasarkan ID sesi.
    config={"configurable": {"session_id": "abc123"}},
)
print(res3)

res2 = chain_with_history.invoke(
    # Memasukkan pertanyaan
    {"question": "Apa nama saya tadi?"},
    # Mencatat percakapan berdasarkan ID sesi.
    config={"configurable": {"session_id": "abc1234"}},
)
print(res2)

