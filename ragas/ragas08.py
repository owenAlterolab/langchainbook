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

from rouge_score import rouge_scorer
from nlp_id.tokenizer import Tokenizer

# Deklarasi tokenizer
id_tokenizer = Tokenizer()

sent1 = "Halo. Senang bertemu. Nama saya Teddy."
sent2 = "Halo, senang bertemu~^^ Nama saya Teddy!!"

# # Tokenisasi
# print(sent1.split())
# print(sent2.split())

# print("===" * 20)

# # Tokenisasi
# print(id_tokenizer.tokenize(sent1))
# print(id_tokenizer.tokenize(sent2))

sent1 = "Halo. Senang bertemu. Nama saya Teddy."
sent2 = "Halo, senang bertemu~^^ Nama saya Teddy!!"
sent3 = "Nama saya Teddy. Halo. Senang bertemu."

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=Tokenizer()
)

# print(
#     f"[1] {sent1}\n[2] {sent2}\n[rouge1] {scorer.score(sent1, sent2)['rouge1'].fmeasure:.5f}\n[rouge2] {scorer.score(sent1, sent2)['rouge2'].fmeasure:.5f}\n[rougeL] {scorer.score(sent1, sent2)['rougeL'].fmeasure:.5f}"
# )
# print("===" * 20)
# print(
#     f"[1] {sent1}\n[2] {sent3}\n[rouge1] {scorer.score(sent1, sent3)['rouge1'].fmeasure:.5f}\n[rouge2] {scorer.score(sent1, sent3)['rouge2'].fmeasure:.5f}\n[rougeL] {scorer.score(sent1, sent3)['rougeL'].fmeasure:.5f}"
# )

from nltk.translate.bleu_score import sentence_bleu

sent1 = "Halo. Senang bertemu. Nama saya Teddy."
sent2 = "Halo, senang bertemu~^^ Nama saya Teddy!!"
sent3 = "Nama saya Teddy. Halo. Senang bertemu."

# tokenize
print("bleu tokenize")
print(id_tokenizer.tokenize(sent1))
print(id_tokenizer.tokenize(sent2))
print(id_tokenizer.tokenize(sent3))

bleu_score = sentence_bleu(
    [id_tokenizer.tokenize(sent1)],
    id_tokenizer.tokenize(sent2),
)
print(f"[1] {sent1}\n[2] {sent2}\n[score] {bleu_score:.5f}")
print("===" * 20)

bleu_score = sentence_bleu(
    [id_tokenizer.tokenize(sent1)],
    id_tokenizer.tokenize(sent3),
)
print(f"[1] {sent1}\n[2] {sent3}\n[score] {bleu_score:.5f}")

print("METEOR")

import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet as wn
wn.ensure_loaded()
from nltk.translate import meteor_score

sent1 = "Halo. Senang bertemu. Nama saya Teddy."
sent2 = "Halo, senang bertemu~^^ Nama saya Teddy!!"
sent3 = "Nama saya Teddy. Halo. Senang bertemu."

meteor = meteor_score.meteor_score(
    [id_tokenizer.tokenize(sent1)],
    id_tokenizer.tokenize(sent2),
)

print(f"[1] {sent1}\n[2] {sent2}\n[score] {meteor:.5f}")
print("===" * 20)

meteor = meteor_score.meteor_score(
    [id_tokenizer.tokenize(sent1)],
    id_tokenizer.tokenize(sent3),
)
print(f"[1] {sent1}\n[2] {sent3}\n[score] {meteor:.5f}")

print("SEMSCORE")

from sentence_transformers import SentenceTransformer, util
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

sent1 = "Halo. Senang bertemu. Nama saya Teddy."
sent2 = "Halo, senang bertemu~^^ Nama saya Teddy!!"
sent3 = "Nama saya Teddy. Halo. Senang bertemu."

# memuat model SentenceTransformer
model = SentenceTransformer("all-mpnet-base-v2")

# encode kalimat-kalimat
sent1_encoded = model.encode(sent1, convert_to_tensor=True)
sent2_encoded = model.encode(sent2, convert_to_tensor=True)
sent3_encoded = model.encode(sent3, convert_to_tensor=True)

# Hitung kemiripan kosinus antara sent1 dan sent2
cosine_similarity = util.pytorch_cos_sim(sent1_encoded, sent2_encoded).item()
print(f"[1] {sent1}\n[2] {sent2}\n[score] {cosine_similarity:.5f}")

print("===" * 20)

# menghitung kemiripan kosinus antara sent1 dan sent3
cosine_similarity = util.pytorch_cos_sim(sent1_encoded, sent3_encoded).item()
print(f"[1] {sent1}\n[2] {sent3}\n[score] {cosine_similarity:.5f}")

from langsmith.schemas import Run, Example
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate import meteor_score
from sentence_transformers import SentenceTransformer, util
import os

# Pengaturan paralelisme tokenizer (menggunakan model HuggingFace)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

smooth = SmoothingFunction()


def rouge_evaluator(metric: str = "rouge1") -> dict:
    # Mendefinisikan fungsi pembungkus
    def _rouge_evaluator(run: Run, example: Example) -> dict:
        # Mengambil output dan jawaban referensi
        student_answer = run.outputs.get("answer", "")
        reference_answer = example.outputs.get("answer", "")

        # Menghitung skor ROUGE
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True, tokenizer=Tokenizer()
        )
        scores = scorer.score(reference_answer, student_answer)

        # Mengembalikan skor ROUGE
        rouge = scores[metric].fmeasure

        return {"key": "ROUGE", "score": rouge}

    return _rouge_evaluator


def bleu_evaluator(run: Run, example: Example) -> dict:
    # Mengambil output dan jawaban referensi
    student_answer = run.outputs.get("answer", "")
    reference_answer = example.outputs.get("answer", "")

    # Tokenisasi
    reference_tokens = id_tokenizer.tokenize(reference_answer)
    student_tokens = id_tokenizer.tokenize(student_answer)

    # Menghitung skor BLEU
    bleu_score = sentence_bleu([reference_tokens], student_tokens, smoothing_function=smooth.method1)

    return {"key": "BLEU", "score": bleu_score}


def meteor_evaluator(run: Run, example: Example) -> dict:
    # Mengambil output dan jawaban referensi
    student_answer = run.outputs.get("answer", "")
    reference_answer = example.outputs.get("answer", "")

    # Tokenisasi
    reference_tokens = id_tokenizer.tokenize(reference_answer)
    student_tokens = id_tokenizer.tokenize(student_answer)

    # Menghitung skor METEOR
    meteor = meteor_score.meteor_score([reference_tokens], student_tokens)

    return {"key": "METEOR", "score": meteor}


def semscore_evaluator(run: Run, example: Example) -> dict:
    try:
        # Mengambil output dan jawaban referensi
        student_answer = run.outputs.get("answer")
        reference_answer = example.outputs.get("answer", "")

        print("semscore")
        print(type(student_answer))
        print(student_answer)

        # memuat model SentenceTransformer
        model = SentenceTransformer("all-mpnet-base-v2")

        #  membuat penyematan kalimat
        student_embedding = model.encode(student_answer, convert_to_tensor=True)
        reference_embedding = model.encode(reference_answer, convert_to_tensor=True)

        print(student_embedding)

        # Menghitung kemiripan kosinus
        cosine_similarity = util.pytorch_cos_sim(
            student_embedding, reference_embedding
        ).item()

        print(cosine_similarity)
    except Exception as e:
        print(f"An error occurred: {e}")
        
    return {"key": "sem_score", "score": cosine_similarity}


from langsmith.evaluation import evaluate

# Mendefinisikan evaluator
heuristic_evalulators = [
    rouge_evaluator(metric="rougeL"),
    bleu_evaluator,
    meteor_evaluator,
    semscore_evaluator,
]

# Menentukan nama dataset
dataset_name = "RAG_EVAL_DATASET"

# Menjalankan eksperimen
experiment_results = evaluate(
    ask_question,
    data=dataset_name,
    evaluators=heuristic_evalulators,
    experiment_prefix="Heuristic-EVAL",
    # Menentukan metadata eksperimen
    metadata={
        "variant": "Evaluasi menggunakan Heuristic-EVAL (Rouge, BLEU, METEOR, SemScore)",
    },
)
