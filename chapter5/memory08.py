from dotenv import load_dotenv
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

# Menginisialisasi model ChatOpenAI.
model = ChatOpenAI()

# Membuat prompt percakapan. Prompt ini mencakup pesan sistem, riwayat percakapan sebelumnya, dan input pengguna.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Anda adalah chatbot yang membantu"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# Buat conversation buffer memory, dan aktifkan fitur pengembalian pesan.
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

print(memory.load_memory_variables({})) # Menginisialisasi variabel-variabel memori ke dalam kamus kosong.

runnable = RunnablePassthrough.assign(
    chat_history = RunnableLambda(memory.load_memory_variables)
    | itemgetter("chat_history") # Masukkan yang sama dengan memory_key.
)

runnable.invoke({"input": "hi"})

chain = runnable | prompt | model

# Menggunakan metode invoke dari objek chain untuk menghasilkan respons terhadap input.
response = chain.invoke({"input": "Senang bertemu dengan Anda. Nama saya Teddy."})
print(response.content)  # Mencetak respons yang dihasilkan.

memory.load_memory_variables({})

# Menyimpan data input dan konten respons ke dalam memori.
memory.save_context(
    {"human": "Senang bertemu dengan Anda. Nama saya Teddy."}, {"ai": response.content}
)

# Mencetak riwayat percakapan yang disimpan.
print(memory.load_memory_variables({}))

# Menanyakan apakah nama masih diingat.
response = chain.invoke({"input": "Apakah Anda ingat apa nama saya?"})
# Mencetak jawaban.
print(response.content)