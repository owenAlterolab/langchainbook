# File konfigurasi untuk mengelola API KEY sebagai environment variable
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

gpt_chain = ask_question_with_llm(ChatOpenAI(model="gpt-4o-mini", temperature=0))


from langsmith.schemas import Run, Example
from langchain_teddynote.evaluator import GroundnessChecker
from langchain_openai import ChatOpenAI

# membuat Pemeriksa Groundness
groundedness_check = GroundnessChecker(
    ChatOpenAI(model="gpt-4o-mini", temperature=0)
).create()

# membuat Pemeriksa Groundness
ollama_groundedness_check = GroundnessChecker(
    ChatOllama(model="llama3.1:8b")
).create()


def openai_groundness_check_evaluator(run: Run, example: Example) -> dict:
    # Jawaban yang dihasilkan LLM, dapatkan jawaban yang benar
    answer = run.outputs.get("answer", "")
    context = run.outputs.get("context", "")

    # Periksa Groundness
    groundedness_score = groundedness_check.invoke(
        {"answer": answer, "context": context}
    )
    groundedness_score = groundedness_score.score == "yes"

    return {"key": "groundness_score", "score": int(groundedness_score)}

def ollama_groundness_check_evaluator(run: Run, example: Example) -> dict:
    # Jawaban yang dihasilkan LLM, dapatkan jawaban yang benar
    answer = run.outputs.get("answer", "")
    context = run.outputs.get("context", "")

    # Periksa Groundness
    groundedness_score = ollama_groundedness_check.invoke(
        {"answer": answer, "context": context}
    )
    groundedness_score = groundedness_score.score == "yes"

    return {"key": "groundness_score", "score": int(groundedness_score)}

from langsmith.evaluation import evaluate

# Menentukan nama dataset
dataset_name = "RAG_EVAL_DATASET"

# Menjalankan evaluasi
# experiment_results = evaluate(
#     gpt_chain,
#     data=dataset_name,
#     evaluators=[
#         teddynote_groundness_check_evaluator,
#     ],
#     experiment_prefix="GROUNDEDNESS-EVAL",
#     # Menentukan metadata eksperimen
#     metadata={
#         "variant": "Evaluasi Halusinasi menggunakan Upstage & TeddyNote Groundness Checker",
#     },
# )



from typing import List
from langsmith.schemas import Example, Run


def ollama_groundness_check_summary_evaluator(
    runs: List[Run], examples: List[Example]
) -> dict:
    def is_grounded(run: Run) -> bool:
        context = run.outputs["context"]
        answer = run.outputs["answer"]
        return (
            ollama_groundedness_check.invoke({"context": context, "answer": answer})
            == "grounded"
        )

    groundedness_scores = sum(1 for run in runs if is_grounded(run))
    return {"key": "groundness_score", "score": groundedness_scores / len(runs)}


def openai_groundness_check_summary_evaluator(
    runs: List[Run], examples: List[Example]
) -> dict:
    def is_grounded(run: Run) -> bool:
        context = run.outputs["context"]
        answer = run.outputs["answer"]
        return (
            groundedness_check.invoke({"context": context, "answer": answer}).score
            == "yes"
        )

    groundedness_scores = sum(1 for run in runs if is_grounded(run))
    return {"key": "groundness_score", "score": groundedness_scores / len(runs)}

from langsmith.evaluation import evaluate

# Menjalankan evaluasi
experiment_result1 = evaluate(
    gpt_chain,
    data=dataset_name,
    summary_evaluators=[
        ollama_groundness_check_summary_evaluator,
    ],
    experiment_prefix="GROUNDNESS_OLLAMA_SUMMARY_EVAL",
    # Menentukan metadata eksperimen
    metadata={
        "variant": "Evaluasi Halusinasi menggunakan Ollama Groundness Checker",
    },
)

# Menjalankan evaluasi
experiment_result2 = evaluate(
    gpt_chain,
    data=dataset_name,
    summary_evaluators=[
        openai_groundness_check_summary_evaluator,
    ],
    experiment_prefix="GROUNDNESS_OPENAI_SUMMARY_EVAL",
    # Menentukan metadata eksperimen
    metadata={
        "variant": "Evaluasi Halusinasi menggunakan Openai Groundness Checker",
    },
)