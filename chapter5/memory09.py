# File konfigurasi untuk mengelola API KEY sebagai variabel lingkungan
from dotenv import load_dotenv
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.utils import ConfigurableFieldSpec


# Memuat informasi API KEY
load_dotenv()
chat_message_history = SQLChatMessageHistory(
    session_id="sql_history", connection="sqlite:///sqlite.db"
)

# Menambahkan pesan dari pengguna.
chat_message_history.add_user_message(
    "Halo? Senang bertemu denganmu. Namaku Teddy. Aku adalah pengembang LangChain. Mari kita bekerja sama dengan baik!"
)
# Menambahkan pesan dari AI.
chat_message_history.add_ai_message("Halo Teddy, senang bertemu denganmu. Mari kita bekerja sama dengan baik!")
print(chat_message_history.messages)

# apply to chain
prompt = ChatPromptTemplate.from_messages(
    [
        # Pesan sistem
        ("system", "Anda adalah asisten yang membantu."),
        # Placeholder untuk riwayat percakapan
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),  # Pertanyaan
    ]
)

# Membuat chain.
chain = prompt | ChatOpenAI(model_name="gpt-4o") | StrOutputParser()

def get_chat_history(user_id, conversation_id):
    return SQLChatMessageHistory(
        table_name=user_id,
        session_id=conversation_id,
        connection="sqlite:///sqlite.db",
    )

config_fields = [
    ConfigurableFieldSpec(
        id="user_id",
        annotation=str,
        name="User ID",
        description="Unique identifier for a user.",
        default="",
        is_shared=True,
    ),
    ConfigurableFieldSpec(
        id="conversation_id",
        annotation=str,
        name="Conversation ID",
        description="Unique identifier for a conversation.",
        default="",
        is_shared=True,
    ),
]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,  # Mengatur fungsi untuk mengambil riwayat percakapan.
    input_messages_key="question",  # Mengatur kunci pesan input sebagai "question".
    history_messages_key="chat_history",  # Mengatur kunci pesan riwayat percakapan sebagai "history".
    history_factory_config=config_fields,  # Mengatur parameter yang akan digunakan saat mengambil riwayat percakapan.
)

# config/pengaturan
config = {"configurable": {"user_id": "user1", "conversation_id": "conversation1"}}

# Masukkan pertanyaan dan konfigurasi untuk dieksekusi.
print(chain_with_history.invoke({"question": "Hai, nama saya Teddy"}, config))

# ajukan pertanyaan lanjutan.
print(chain_with_history.invoke({"question": "Siapa nama saya?"}, config))

# Mengatur konfigurasi
config = {"configurable": {"user_id": "user1", "conversation_id": "conversation2"}}

# Menjalankan dengan mengirimkan pertanyaan dan konfigurasi.
print(chain_with_history.invoke({"question": "Apa nama saya tadi?"}, config))