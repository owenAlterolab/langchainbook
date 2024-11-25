from dotenv import load_dotenv

# Memuat informasi API KEY
load_dotenv()

from langchain_altero import logging

# Masukkan nama proyek
logging.langsmith("CH15-Evaluations")

from myrag import PDFRAG

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

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

# Menggunakan LLM untuk membuat rantai pertanyaan dengan GPT
gpt_chain = ask_question_with_llm(ChatOpenAI(model="gpt-4o-mini", temperature=1.0))

# Memuat model Ollama
ollama_chain = ask_question_with_llm(
    ChatOllama(model="llama3.1:8b", temperature=1.0)
)

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
evaluate(
    gpt_chain,
    data=dataset_name,
    evaluators=[cot_qa_evalulator],
    experiment_prefix="REPEAT_EVAL",
    # Menentukan metadata eksperimen
    metadata={
        "variant": "Melakukan evaluasi berulang. Model GPT-4o-mini (cot_qa)",
    },
    num_repetitions=3,
)

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
evaluate(
    ollama_chain,
    data=dataset_name,
    evaluators=[cot_qa_evalulator],
    experiment_prefix="REPEAT_EVAL",
    # Menentukan metadata eksperimen
    metadata={
        "variant": "Melakukan evaluasi berulang. Model Llama3.1 (cot_qa)",
    },
    num_repetitions=3,
)
