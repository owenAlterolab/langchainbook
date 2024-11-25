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
ollama_chain = ask_question_with_llm(ChatOllama(model="llama3.1:8b"))


from langchain_teddynote.evaluator import OpenAIRelevanceGrader
from langchain_openai import ChatOpenAI


rq_grader = OpenAIRelevanceGrader(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0), target="retrieval-question"
).create()

ra_grader = OpenAIRelevanceGrader(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0), target="retrieval-answer"
).create()

# ress = rq_grader.invoke(
#     {
#         "input": "AI generatif yang dikembangkan oleh Samsung adalah Bixby.",
#         "context": "AI Samsung adalah Bixby",
#     }
# )

# print(ress)

# ress = ra_grader.invoke(
#     {
#         "input": "AI generatif yang dikembangkan oleh Samsung adalah Gauss.",
#         "context": "AI Samsung adalah Bixby",
#     }
# )
# print(ress)

from typing import List
from langsmith.schemas import Example, Run
from langsmith.evaluation import evaluate

def relevance_score_summary_evaluator(runs: List[Run], examples: List[Example]) -> dict:
    rq_scores = 0  # Skor relevansi pertanyaan
    ra_scores = 0  # Skor relevansi jawaban

    for run, example in zip(runs, examples):
        question = example.inputs["question"]
        context = run.outputs["context"]
        prediction = run.outputs["answer"]

        # Evaluasi relevansi pertanyaan
        rq_score = rq_grader.invoke(
            {
                "input": question,
                "context": context,
            }
        )
        # Evaluasi relevansi jawaban
        ra_score = ra_grader.invoke(
            {
                "input": prediction,
                "context": context,
            }
        )

        # Akumulasi skor relevansi
        if rq_score.score == "yes":
            rq_scores += 1
        if ra_score.score == "yes":
            ra_scores += 1

    # Menghitung skor relevansi akhir (rata-rata relevansi pertanyaan dan jawaban)
    final_score = ((rq_scores / len(runs)) + (ra_scores / len(runs))) / 2

    return {"key": "relevance_score", "score": final_score}

# Menjalankan evaluasi
dataset_name = "RAG_EVAL_DATASET"

experiment_result1 = evaluate(
    gpt_chain,
    data=dataset_name,
    summary_evaluators=[relevance_score_summary_evaluator],
    experiment_prefix="SUMMARY_EVAL",
    # Menentukan metadata eksperimen
    metadata={
        "variant": "Menggunakan GPT-4o-mini: Evaluasi relevansi dengan summary_evaluator",
    },
)

# Menjalankan evaluasi
experiment_result2 = evaluate(
    ollama_chain,
    data=dataset_name,
    summary_evaluators=[relevance_score_summary_evaluator],
    experiment_prefix="SUMMARY_EVAL",
    # Menentukan metadata eksperimen
    metadata={
        "variant": "Menggunakan Ollama(Llama3.1:8b): Evaluasi relevansi dengan summary_evaluator",
    },
)