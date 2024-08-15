# File konfigurasi untuk mengelola API KEY sebagai variabel lingkungan
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

# Memuat informasi API KEY
load_dotenv()

print(ENTITY_MEMORY_CONVERSATION_TEMPLATE.template)

# Membuat sebuah LLM.
llm = ChatOpenAI( temperature = 0 )

# Membuat sebuah Rantai Percakapan.
conversation = ConversationChain(
    llm = llm,
    prompt = ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory = ConversationEntityMemory (llm = llm),
)

conversation.predict(
    input="Teddy dan Shirley adalah rekan kerja di sebuah perusahaan. "
          "Teddy adalah seorang pengembang dan Shirley adalah seorang desainer. "
          "Mereka baru saja berhenti dari pekerjaan mereka di perusahaan dan berencana untuk memulai perusahaan mereka sendiri."
)

print(conversation.memory.entity_store.store)