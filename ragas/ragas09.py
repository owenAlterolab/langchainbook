# File konfigurasi untuk mengelola API KEY sebagai environment variable
from dotenv import load_dotenv

# Memuat informasi API KEY
load_dotenv()

from langchain_altero import logging

# Masukkan nama proyek
logging.langsmith("CH15-Evaluations")

from myrag import PDFRAG
from langchain_openai import ChatOpenAI

# Membuat fungsi untuk menjawab pertanyaan menggunakan LLM
def ask_question_with_llm(llm):
    # Membuat objek PDFRAG
    rag = PDFRAG(
        "ragas/data/ChatGPT:Keuntungan,Risiko,DanPenggunaanBijakDalamEraKecerdasanBuatan.pdf",
        llm,
    )

    # Membuat retriever
    retriever = rag.create_retriever()

    # Membuat chain
    rag_chain = rag.create_chain(retriever)

    def _ask_question(inputs: dict):
        context = retriever.invoke(inputs["question"])
        context = "\n".join([doc.page_content for doc in context])
        return {
            "question": inputs["question"],
            "context": context,
            "answer": rag_chain.invoke(inputs["question"]),
        }

    return _ask_question

from langchain_community.chat_models import ChatOllama

# Mengimpor model Ollama.
ollama = ChatOllama(model="llama3.1:8b")

# Memanggil model Ollama
ress = ollama.invoke("hello?")

print(ress)

gpt_chain = ask_question_with_llm(ChatOpenAI(model="gpt-4o-mini", temperature=0))
ollama_chain = ask_question_with_llm(ChatOllama(model="llama3.1:8b"))

from langsmith.evaluation import evaluate, LangChainStringEvaluator

# Membuat evaluator QA
cot_qa_evalulator = LangChainStringEvaluator(
    "cot_qa",
    config={"llm": ChatOpenAI(model="gpt-4o-mini", temperature=0)},
    prepare_data=lambda run, example: {
        "prediction": run.outputs["answer"],
        "reference": run.outputs["context"],
        "input": example.inputs["question"],
    },
)

dataset_name = "RAG_EVAL_DATASET"

# Menjalankan evaluasi
experiment_results1 = evaluate(
    gpt_chain,
    data=dataset_name,
    evaluators=[cot_qa_evalulator],
    experiment_prefix="MODEL_COMPARE_EVAL",
    # Menentukan metadata eksperimen
    metadata={
        "variant": "Evaluasi GPT-4o-mini (cot_qa)",
    },
)

# Menjalankan evaluasi
experiment_results2 = evaluate(
    ollama_chain,
    data=dataset_name,
    evaluators=[cot_qa_evalulator],
    experiment_prefix="MODEL_COMPARE_EVAL",
    # Menentukan metadata eksperimen
    metadata={
        "variant": "Evaluasi Ollama(llama3.1:8b) (cot_qa)",
    },
)