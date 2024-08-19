from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

load_dotenv()
# memory = ConversationBufferMemory()
# Mengatur return_messages = True akan mengembalikan objek HumanMessage dan AIMessage.
memory = ConversationBufferMemory(return_messages=False)

memory.save_context(
    inputs = {
        "human": "Hai, saya ingin membuka rekening bank secara tatap muka, bagaimana cara memulainya?"
    },
    outputs = {
        "ai": "Halo, kami senang mendengar Anda ingin membuka rekening. Pertama, bisakah Anda menyiapkan kartu identitas untuk memverifikasi identitas Anda?"
    },
)

print(memory.load_memory_variables({}))

memory.save_context(
    inputs={"human": "Ya, saya memiliki ID saya, sekarang apa yang harus saya lakukan?"},
    outputs = {
        "ai": "Terima kasih. Silakan ambil foto yang jelas dari bagian depan dan belakang kartu identitas Anda dan unggah, dan kami akan melanjutkan untuk memverifikasi identitas Anda."
    },
)

memory.save_context(
    inputs = {"manusia": "Saya telah mengunggah foto, bagaimana cara memverifikasi identitas saya?"},
    outputs = {
        "ai": "Kami telah memverifikasi foto yang Anda unggah, sekarang silakan verifikasi identitas Anda melalui ponsel Anda. Silakan masukkan nomor verifikasi yang dikirimkan kepada Anda melalui SMS."
    },
)
memory.save_context(
    inputs = {"manusia": "Saya sudah memasukkan nomor verifikasi, sekarang bagaimana cara membuka akun?" },
    outputs = {
        "ai": "Identitas Anda telah diverifikasi, sekarang silakan pilih jenis akun yang Anda inginkan dan isi informasi yang diperlukan. Anda bisa memilih jenis deposit, mata uang, dll."
    },
)

print(memory.load_memory_variables({})["history"])

#conversation chain
llm = ChatOpenAI(temperature=0)

conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
)

# Memulai dialog.
response = conversation.predict(
    input="Hai, saya ingin membuka rekening bank secara tatap muka, bagaimana cara memulainya?"
)
print(response)

response = conversation.predict(
    input = "Tolong rangkum jawaban Anda sebelumnya ke dalam poin-poin."
)
print(response)