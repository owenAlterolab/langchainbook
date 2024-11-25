from dotenv import load_dotenv
load_dotenv()


from langchain_altero import logging
logging.langsmith("CH15-Evaluations")

from myrag import PDFRAG
from langchain_openai import ChatOpenAI

# Membuat objek PDFRAG
rag = PDFRAG(
    "ragas/data/ChatGPT:Keuntungan,Risiko,DanPenggunaanBijakDalamEraKecerdasanBuatan.pdf",
    ChatOpenAI(model="gpt-4o-mini", temperature=0),
)

# Membuat retriever
retriever = rag.create_retriever()

# Membuat chain
chain = rag.create_chain(retriever)

# # Menghasilkan jawaban untuk pertanyaan
# chain.invoke("Apa risiko utama dalam penggunaan ChatGPT?")

# Membuat fungsi untuk menjawab pertanyaan
def ask_question(inputs: dict):
    return {"answer": chain.invoke(inputs["question"])}

from langsmith.schemas import Run, Example
import random


def random_score_evaluator(run: Run, example: Example) -> dict:
    # mengembalikan skor acak
    score = random.randint(1, 11)
    return {"key": "random_score", "score": score}

from langsmith.evaluation import evaluate

# mengatur nama dataset
dataset_name = "RAG_EVAL_DATASET"

# jalankan
# experiment_results = evaluate(
#     ask_question,
#     data=dataset_name,
#     evaluators=[random_score_evaluator],
#     experiment_prefix="CUSTOM-EVAL",
#     # tentukan metadata
#     metadata={
#         "variant": "evaluator skor acak",
#     },
# )

# Fungsi pengembalian hasil RAG yang mengembalikan Konteks
def context_answer_rag_answer(inputs: dict):
    context = retriever.invoke(inputs["question"])
    return {
        "context": "\n".join([doc.page_content for doc in context]),
        "answer": chain.invoke(inputs["question"]),
        "question": inputs["question"],
    }

from langchain import hub

# Dapatkan evaluator promp
llm_evaluator_prompt = hub.pull("teddynote/context-answer-evaluator")
# llm_evaluator_prompt.pretty_print()

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Membuat evaluator
custom_llm_evaluator = (
    llm_evaluator_prompt
    | ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
    | StrOutputParser()
)

# Menghasilkan jawaban.
output = context_answer_rag_answer(
    {"question": "Apa risiko utama dalam penggunaan ChatGPT?"}
)

# Menjalankan evaluasi skor
res = custom_llm_evaluator.invoke(output)
print(res)

from langsmith.schemas import Run, Example


def custom_evaluator(run: Run, example: Example) -> dict:
    # Jawaban yang dihasilkan LLM, dapatkan jawaban yang benar.
    llm_answer = run.outputs.get("answer", "")
    context = run.outputs.get("context", "")
    question = example.outputs.get("question", "")

    # kembalikan skor acak
    score = custom_llm_evaluator.invoke(
        {"question": question, "answer": llm_answer, "context": context}
    )
    return {"key": "custom_score", "score": float(score)}

from langsmith.evaluation import evaluate

# Menentukan nama dataset
dataset_name = "RAG_EVAL_DATASET"

# Menjalankan evaluasi
experiment_results = evaluate(
    context_answer_rag_answer,
    data=dataset_name,
    evaluators=[custom_evaluator],
    experiment_prefix="CUSTOM-LLM-EVAL2",
    # Menentukan metadata eksperimen
    metadata={
        "variant": "Evaluasi menggunakan Custom LLM Evaluator",
    },
)

