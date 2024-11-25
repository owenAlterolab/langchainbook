# File konfigurasi untuk mengelola API KEY sebagai environment variable
from dotenv import load_dotenv

# Memuat informasi API KEY
load_dotenv()

from langchain_altero import logging

# Masukkan nama proyek
logging.langsmith("CH15-Evaluations")

from langchain import hub

from langchain_openai import ChatOpenAI
from langsmith.schemas import Example, Run
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langsmith.evaluation import evaluate



def evaluate_pairwise(runs: list, example) -> dict:
    """
    Evaluator sederhana untuk jawaban pasangan yang memberikan skor berdasarkan keterlibatan
    """

    # Menyimpan skor
    scores = {}
    for i, run in enumerate(runs):
        scores[run.id] = i

    # Pasangan eksekusi untuk setiap contoh
    answer_a = runs[0].outputs["answer"]
    answer_b = runs[1].outputs["answer"]
    question = example.inputs["question"]

    # Menggunakan LLM dengan pemanggilan fungsi, model dengan kinerja terbaik
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Prompt terstruktur
    grade_prompt = PromptTemplate.from_template(
        """
        Anda adalah juri LLM. Bandingkan kedua jawaban berikut untuk suatu pertanyaan dan tentukan mana yang lebih baik.
        Jawaban yang lebih baik adalah yang lebih rinci dan informatif.
        Jika jawaban tidak terkait dengan pertanyaan, itu bukan jawaban yang baik.

        # Pertanyaan:
        {question}

        # Jawaban A: 
        {answer_a}

        # Jawaban B: 
        {answer_b}

        Output harus berupa `A` atau `B`. Pilih jawaban yang lebih baik.

        #Preferensi:
        """
    )
    answer_grader = grade_prompt | llm | StrOutputParser()

    # Mendapatkan skor
    score = answer_grader.invoke(
        {
            "question": question,
            "answer_a": answer_a,
            "answer_b": answer_b,
        }
    )
    # score = score["Preference"]

    # Pemetaan alokasi eksekusi berdasarkan skor
    if score == "A":  # Preferensi Asisten A
        scores[runs[0].id] = 1
        scores[runs[1].id] = 0
    elif score == "B":  # Preferensi Asisten B
        scores[runs[0].id] = 0
        scores[runs[1].id] = 1
    else:
        scores[runs[0].id] = 0
        scores[runs[1].id] = 0

    return {"key": "ranked_preference", "scores": scores}



from langsmith.evaluation import evaluate_comparative

# Mengganti array nama atau ID eksperimen
evaluate_comparative(
    ["GROUNDNESS_OPENAI_SUMMARY_EVAL-379e50f7", "GROUNDNESS_OPENAI_SUMMARY_EVAL-c7e4a182"],
    # Array evaluator
    evaluators=[evaluate_pairwise],
)