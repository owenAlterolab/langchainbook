from dotenv import load_dotenv
from langchain_altero import logging
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load API key information
load_dotenv()

# Enter the project name.
logging.langsmith("LCEL-Advanced")

runnable = RunnableParallel(
    # Set up a Runnable that returns the passed input as is.
    passed=RunnablePassthrough(),
    # Set up a Runnable that returns the result of multiplying the "num" value of the input by 3.
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    # Set up a Runnable that returns the result of adding 1 to the "num" value of the input.
    modified=lambda x: x["num"] + 1,
)

# Run the Runnable with {"num": 1} as input.
invoke_result = runnable.invoke({"num": 1})
print(invoke_result)

r = RunnablePassthrough.assign(mult=lambda x: x["num"] * 3)
invoke_result = r.invoke({"num": 1})
print()
print(invoke_result)

# Membuat FAISS vector store dari teks.
vectorstore = FAISS.from_texts(
    [
        "Teddy bekerja di LangChain Inc.",
        "Shirley bekerja di perusahaan yang sama dengan Teddy.",
        "Pekerjaan Teddy adalah pengembang.",
        "Pekerjaan Shirley adalah desainer.",
    ],
    embedding=OpenAIEmbeddings(),
)
# Menggunakan vector store sebagai retriever.
retriever = vectorstore.as_retriever()
# Membuat template.
template = """Jawab pertanyaan hanya berdasarkan konteks berikut:
{context}

Pertanyaan: {question}
"""
# Membuat chat prompt dari template.
prompt = ChatPromptTemplate.from_template(template)

# Menginisialisasi model ChatOpenAI.
model = ChatOpenAI(model_name="gpt-4o-mini")

# Fungsi untuk memformat dokumen
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

# Membuat rantai pengambilan.
retrieval_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Memanggil rantai pengambilan untuk mendapatkan jawaban dari pertanyaan.
teddy = retrieval_chain.invoke("Apa pekerjaan Teddy?")
print(teddy)

print()

shirley = retrieval_chain.invoke("Apa pekerjaan Shirley?")
print(shirley)