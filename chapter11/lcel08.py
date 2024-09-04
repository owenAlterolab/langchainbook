from dotenv import load_dotenv
from langchain_altero import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableParallel
import os

# Load API key information
load_dotenv()

# Enter the project name.
logging.langsmith("LCEL-Advanced")

# Tetapkan variabel lingkungan LANGCHAIN_TRACING_V2 ke “true”.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# Mengatur LANGCHAIN_PROJECT
os.environ["LANGCHAIN_PROJECT"] = "RunnableWithMessageHistory"

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Anda adalah asisten yang mahir dalam {ability}. Harap jawab dalam 20 karakter atau kurang",
        ),
        # menggunakan riwayat percakapan sebagai variabel, dengan riwayat sebagai kunci dari MessageHistory
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"), # gunakan masukan pengguna sebagai variabel
    ]
)
runnable = prompt | model # menggabungkan prompt dan model untuk membuat objek yang dapat dijalankan

store = {}  # Kamus untuk menyimpan riwayat sesi


# Fungsi untuk mengambil riwayat sesi berdasarkan ID sesi
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    print(session_ids)
    if session_ids not in store:  # Jika ID sesi tidak ada di store
        # Buat objek ChatMessageHistory baru dan simpan ke store
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # Kembalikan riwayat sesi untuk ID sesi tersebut


with_message_history = (
    RunnableWithMessageHistory(  # Buat objek RunnableWithMessageHistory
        runnable,  # Objek Runnable yang akan dijalankan
        get_session_history,  # Fungsi untuk mengambil riwayat sesi
        input_messages_key="input",  # Kunci untuk pesan input
        history_messages_key="history",  # Kunci untuk pesan riwayat
    )
)

history_factory_config = None

if history_factory_config:
    _config_specs = history_factory_config
else:
    # If not provided, then we'll use the default session_id field
    _config_specs = [
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="Unique identifier for a session.",
            default="",
            is_shared=True,
        ),
    ]

inv = with_message_history.invoke(
    # Mengirimkan pertanyaan terkait matematika "Apa arti dari kosinus?" sebagai input.
    {"ability": "math", "input": "Apa itu kosinus?"},
    # Mengirimkan ID sesi "abc123" sebagai informasi konfigurasi.
    config={"configurable": {"session_id": "abc123"}},
)
print(inv)

# Memanggil dengan menyertakan riwayat pesan.
inv = with_message_history.invoke(
    # Menetapkan kemampuan dan input.
    {"ability": "math", "input": "Tolong jawab dalam bahasa Inggris pertanyaan sebelumnya."},
    # Menetapkan opsi konfigurasi.
    config={"configurable": {"session_id": "abc123"}},
)
print(inv)

# Karena session_id baru, tidak mengingat percakapan sebelumnya.
inv = with_message_history.invoke(
    # Mengirimkan kemampuan matematika dan pesan input.
    {"ability": "math", "input": "Tolong jawab dalam bahasa Inggris pertanyaan sebelumnya"},
    # Menetapkan session_id baru.
    config={"configurable": {"session_id": "def234"}},
)
print(inv)

store = {}  # Inisialisasi kamus kosong.


def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    # Mengembalikan riwayat sesi yang sesuai dengan user_id dan conversation_id yang diberikan.
    if (user_id, conversation_id) not in store:
        # Jika kunci tersebut tidak ada di store, buat dan simpan ChatMessageHistory baru.
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[  # Ini akan menggantikan pengaturan "session_id" yang ada.
        ConfigurableFieldSpec(
            id="user_id",  # Digunakan sebagai argumen pertama dari fungsi get_session_history.
            annotation=str,
            name="User ID",
            description="Pengenal unik untuk pengguna.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",  # Digunakan sebagai argumen kedua dari fungsi get_session_history.
            annotation=str,
            name="Conversation ID",
            description="Pengenal unik untuk percakapan.",
            default="",
            is_shared=True,
        ),
    ],
)

inv = with_message_history.invoke(
    # Mengirimkan kamus yang berisi kemampuan (ability) dan input (input).
    {"ability": "math", "input": "Hello"},
    # Mengirimkan kamus konfigurasi (config).
    config={"configurable": {"user_id": "123", "conversation_id": "1"}},
)
print(inv)

# Membuat chain
chain = RunnableParallel({"output_message": ChatOpenAI()})


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # Jika riwayat percakapan untuk session ID tidak ada di penyimpanan, buat ChatMessageHistory baru.
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    # Kembalikan riwayat percakapan untuk session ID yang diberikan.
    return store[session_id]


# Membuat objek RunnableWithMessageHistory yang menambahkan fungsi riwayat percakapan ke dalam chain.
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    # Mengatur kunci pesan input ke "input". (Diabaikan jika input berupa objek Message)
    # input_messages_key="input",
    # Mengatur kunci pesan output ke "output_message". (Diabaikan jika output berupa objek Message)
    output_messages_key="output_message",
)

# Menjalankan chain dengan pesan dan konfigurasi yang diberikan.
inv = with_message_history.invoke(
    # Atau bisa juga menggunakan "what is the definition of cosine?"
    [HumanMessage(content="Apa definisi dari kosinus?")],
    config={"configurable": {"session_id": "abc123"}},
)
print(inv)

inv = with_message_history.invoke(
    # Meminta ulang jawaban sebelumnya dalam bahasa Korea.
    [HumanMessage(content="Tolong jawab dalam bahasa Korea Inggris pertanyaan sebelumnya!")],
    # Mengirimkan opsi konfigurasi dalam bentuk kamus.
    config={"configurable": {"session_id": "abc123"}},
)
print(inv)

with_message_history = RunnableWithMessageHistory(
    ChatOpenAI(),  # Menggunakan model bahasa ChatOpenAI.
    get_session_history,  # Menetapkan fungsi untuk mengambil riwayat sesi percakapan.
    # Mengatur kunci pesan input ke "input". (Diabaikan jika input berupa objek Message)
    # input_messages_key="input",
    # Mengatur kunci pesan output ke "output_message". (Diabaikan jika output berupa objek Message)
    # output_messages_key="output_message",
)
inv = with_message_history.invoke(
    # Meminta ulang jawaban dalam bahasa Korea berdasarkan percakapan sebelumnya.
    [HumanMessage(content="Apa arti dari kosinus?")],
    # Mengirimkan opsi konfigurasi dalam bentuk kamus.
    config={"configurable": {"session_id": "def123"}},
)
print(inv)

from operator import itemgetter

with_message_history = RunnableWithMessageHistory(
    # Menggunakan kunci "input_messages" untuk mengambil pesan input dan mengirimkannya ke ChatOpenAI().
    itemgetter("input_messages") | ChatOpenAI(),
    get_session_history,  # Fungsi untuk mengambil riwayat sesi.
    input_messages_key="input_messages",  # Menetapkan kunci untuk pesan input.
)
inv = with_message_history.invoke(
    {"input_messages": "Apa arti dari kosinus?"},
    # Mengirimkan opsi konfigurasi dalam bentuk kamus.
    config={"configurable": {"session_id": "xyz123"}},
)
print(inv)

from langchain_community.chat_message_histories import RedisChatMessageHistory

REDIS_URL = "redis://localhost:6379/0"

def get_message_history(session_id: str) -> RedisChatMessageHistory:
    # Mengembalikan objek RedisChatMessageHistory berdasarkan session ID.
    return RedisChatMessageHistory(session_id, url=REDIS_URL)


with_message_history = RunnableWithMessageHistory(
    runnable,  # Objek yang dapat dijalankan
    get_message_history,  # Fungsi untuk mengambil riwayat pesan
    input_messages_key="input",  # Kunci untuk pesan input
    history_messages_key="history",  # Kunci untuk pesan riwayat
)

inv = with_message_history.invoke(
    # Mengirimkan pertanyaan terkait matematika "Apa arti dari kosinus?" sebagai input.
    {"ability": "math", "input": "Apa arti dari kosinus?"},
    # Menetapkan session ID menjadi "redis123" sebagai opsi konfigurasi.
    config={"configurable": {"session_id": "redis123"}},
)
print(inv)

inv = with_message_history.invoke(
    # Meminta terjemahan jawaban sebelumnya ke dalam bahasa Korea.
    {"ability": "math", "input": "Tolong terjemahkan jawaban sebelumnya ke dalam bahasa Inggris."},
    # Menetapkan session ID menjadi "redis123" sebagai nilai konfigurasi.
    config={"configurable": {"session_id": "redis123"}},
)
print(inv)

inv = with_message_history.invoke(
    # Meminta terjemahan jawaban sebelumnya ke dalam bahasa Korea.
    {"ability": "math", "input": "Tolong terjemahkan jawaban sebelumnya ke dalam bahasa Inggris."},
    # Menetapkan session ID menjadi "redis456" sebagai nilai konfigurasi.
    config={"configurable": {"session_id": "redis456"}},
)
print(inv)