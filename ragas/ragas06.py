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

from langsmith.evaluation import LangChainStringEvaluator
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import os

# Pengaturan paralelisme tokenizer (menggunakan model HuggingFace)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

model_name = "BAAI/bge-m3"

hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},  # cuda, cpu
    # encode_kwargs={"normalize_embeddings": True},
)

# Membuat evaluator model embedding
hf_embedding_evaluator = LangChainStringEvaluator(
    "embedding_distance",
    config={
        # OpenAIEmbeddings diatur sebagai nilai default, tetapi dapat diubah
        "embeddings": hf_embeddings,
        "distance_metric": "cosine",  # "cosine", "euclidean", "chebyshev", "hamming", dan "manhattan"
    },
)

openai_embedding_evaluator = LangChainStringEvaluator(
    "embedding_distance",
    config={
        # OpenAIEmbeddings diatur sebagai nilai default, tetapi dapat diubah
        "embeddings": OpenAIEmbeddings(model="text-embedding-3-small"),
        "distance_metric": "euclidean",  # "cosine", "euclidean", "chebyshev", "hamming", dan "manhattan"
    },
)

from langsmith.evaluation import evaluate

dataset_name = "RAG_EVAL_DATASET"

# jalankan evaluasi
experiment_results = evaluate(
    ask_question,
    data=dataset_name,
    evaluators=[
        hf_embedding_evaluator,
        openai_embedding_evaluator
    ],
    experiment_prefix="EMBEDDING-EVAL",
    # tentukan metadata evaluasi
    metadata={
        "variant": "Evaluasi menggunakan embedding_distance",
    },
)
