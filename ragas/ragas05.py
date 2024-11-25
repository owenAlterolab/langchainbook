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

# Menghasilkan jawaban untuk pertanyaan
# res = chain.invoke("Apa risiko utama dalam penggunaan ChatGPT?")
# print(res)

# Membuat fungsi dengan nama ask_question
def ask_question(inputs: dict):
    return {"answer": chain.invoke(inputs["question"])}

# Contoh pertanyaan dari pengguna
# llm_answer = ask_question(
#     {"question": "Apa risiko utama dalam penggunaan ChatGPT?"}
# )
# print(llm_answer)

# Fungsi untuk keluaran evaluator
def print_evaluator_prompt(evaluator):
    return evaluator.evaluator.prompt.pretty_print()

from langsmith.evaluation import evaluate, LangChainStringEvaluator

# membuat evaluator qa
qa_evalulator = LangChainStringEvaluator("qa")

# Mencetak prompt
# print_evaluator_prompt(qa_evalulator)

dataset_name = "RAG_EVAL_DATASET"

# # Jalankan evaluasi
# experiment_results = evaluate(
#     ask_question,
#     data=dataset_name,
#     evaluators=[qa_evalulator],
#     experiment_prefix="RAG_EVAL",
#     # tentukan metadata percobaan
#     metadata={
#         "variant": "Evaluasi menggunakan QA Evaluator",
#     },
# )

# Fungsi pengembalian hasil RAG yang mengembalikan Konteks
def context_answer_rag_answer(inputs: dict):
    context = retriever.invoke(inputs["question"])
    return {
        "context": "\n".join([doc.page_content for doc in context]),
        "answer": chain.invoke(inputs["question"]),
        "query": inputs["question"],
    }

# Menjalankan fungsi
# answ = context_answer_rag_answer(
#     {"question": "Apa risiko utama dalam penggunaan ChatGPT?"}
# )

# print(answ)

# membuat cot_qa evaluator
cot_qa_evaluator = LangChainStringEvaluator(
    "cot_qa",
    prepare_data=lambda run, example: {
        "prediction": run.outputs["answer"],  # jawaban yang dihasilkan oleh LLM
        "reference": run.outputs["context"],  # Konteks
        "input": example.inputs["question"],  # Pertanyaan dari dataset
    },
)

# Membuat evaluator context_qa
context_qa_evaluator = LangChainStringEvaluator(
    "context_qa",
    prepare_data=lambda run, example: {
        "prediction": run.outputs["answer"],  # jawaban yang dihasilkan oleh LLM
        "reference": run.outputs["context"],  # Konteks
        "input": example.inputs["question"],  # Pertanyaan dari dataset
    },
)

# cetak prompt evaluator
# print_evaluator_prompt(context_qa_evaluator)

# Menentukan nama dataset
dataset_name = "RAG_EVAL_DATASET"

# # Menjalankan evaluasi
# evaluate(
#     context_answer_rag_answer,
#     data=dataset_name,
#     evaluators=[cot_qa_evaluator, context_qa_evaluator],
#     experiment_prefix="RAG_EVAL",
#     metadata={
#         "variant": "Evaluasi menggunakan COT_QA & Context_QA Evaluator",
#     },
# )

from langsmith.evaluation import evaluate, LangChainStringEvaluator

# menyiapkan evaluator
criteria_evaluator = [
    LangChainStringEvaluator("criteria", config={"criteria": "conciseness"}),
    LangChainStringEvaluator("criteria", config={"criteria": "misogyny"}),
    LangChainStringEvaluator("criteria", config={"criteria": "criminality"}),
]

# mengatur nama dataset
dataset_name = "RAG_EVAL_DATASET"

# jalankan evaluasi
# experiment_results = evaluate(
#     ask_question,
#     data=dataset_name,
#     evaluators=criteria_evaluator,
#     experiment_prefix="CRITERIA-EVAL",
#     # tentukan metadata percobaan
#     metadata={
#         "variant": "Evaluasi menggunakan kriteria",
#     },
# )

from langsmith.evaluation import LangChainStringEvaluator
from langchain_openai import ChatOpenAI

# labeled_criteria 평가자 생성
labeled_criteria_evaluator = LangChainStringEvaluator(
    "labeled_criteria",
    config={
        "criteria": {
            "helpfulness": (
                "Is this submission helpful to the user,"
                " taking into account the correct reference answer?"
            )
        },
        "llm": ChatOpenAI(temperature=0.0, model="gpt-4o-mini"),
    },
    prepare_data=lambda run, example: {
        "prediction": run.outputs["answer"],
        "reference": example.outputs["answer"],  # 정답 답변
        "input": example.inputs["question"],
    },
)

# print("labeled_criteria_evaluator")
# evaluator prompt 출력
# print_evaluator_prompt(labeled_criteria_evaluator)

from langchain_openai import ChatOpenAI

relevance_evaluator = LangChainStringEvaluator(
    "labeled_criteria",
    config={
        "criteria": "relevance",
        "llm": ChatOpenAI(temperature=0.0, model="gpt-4o-mini"),
    },
    prepare_data=lambda run, example: {
        "prediction": run.outputs["answer"],
        "reference": run.outputs["context"],  # Berikan context
        "input": example.inputs["question"],
    },
)

# print("relevance_evaluator")
# print_evaluator_prompt(relevance_evaluator)

from langsmith.evaluation import evaluate

# Menentukan nama dataset
dataset_name = "RAG_EVAL_DATASET"

# Menjalankan evaluasi
# experiment_results = evaluate(
#     context_answer_rag_answer,
#     data=dataset_name,
#     evaluators=[labeled_criteria_evaluator, relevance_evaluator],
#     experiment_prefix="LABELED-EVAL",
#     # Menentukan metadata eksperimen
#     metadata={
#         "variant": "Evaluasi menggunakan labeled_criteria evaluator",
#     },
# )

from langsmith.evaluation import LangChainStringEvaluator

# Membuat evaluator yang mengembalikan skor
labeled_score_evaluator = LangChainStringEvaluator(
    "labeled_score_string",
    config={
        "criteria": {
            "accuracy": "How accurate is this prediction compared to the reference on a scale of 1-10?"
        },
        "normalize_by": 10,
        "llm": ChatOpenAI(temperature=0.0, model="gpt-4o-mini"),
    },
    prepare_data=lambda run, example: {
        "prediction": run.outputs["answer"],
        "reference": example.outputs["answer"],
        "input": example.inputs["question"],
    },
)

# print_evaluator_prompt(labeled_score_evaluator)

from langsmith.evaluation import evaluate

# jalankan evaluasi
experiment_results = evaluate(
    ask_question,
    data=dataset_name,
    evaluators=[labeled_score_evaluator],
    experiment_prefix="LABELED-SCORE-EVAL",
    # Tentukan metadata percobaan
    metadata={
        "variant": "Evaluasi menggunakan labelled_score",
    },
)
