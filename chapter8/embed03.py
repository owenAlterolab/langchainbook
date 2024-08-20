from dotenv import load_dotenv
import os
import warnings
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings
import numpy as np


load_dotenv()
warnings.filterwarnings("ignore")

texts = [
    "Halo, senang bertemu denganmu.",
    "LangChain menyederhanakan proses membangun aplikasi dengan model bahasa besar.",
    "Tutorial LangChain dalam bahasa Korea disusun berdasarkan dokumentasi resmi LangChain, cookbook, dan berbagai contoh praktis untuk membantu pengguna memanfaatkan LangChain dengan lebih mudah dan efektif.",
    "LangChain menyederhanakan proses pembangunan aplikasi menggunakan model bahasa besar.",
    "Retrieval-Augmented Generation (RAG) adalah teknik efektif untuk meningkatkan respons AI.",
]

model_name = "intfloat/multilingual-e5-large-instruct"

hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=model_name,
    task="feature-extraction",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)

embedded_documents = hf_embeddings.embed_documents(texts)
embedded_query = hf_embeddings.embed_query("Tolong beri tahu saya tentang LangChain.")

print("[HuggingFace Endpoint Embedding]")
print(f"Model: \t\t{model_name}")
print(f"Dimension: \t{len(embedded_documents[0])}")

np.array(embedded_query) @ np.array(embedded_documents).T

sorted_idx = (np.array(embedded_query) @ np.array(embedded_documents).T).argsort()[::-1]
sorted_idx

print("[Query] Tolong beri tahu saya tentang LangChain.\n====================================")
for i, idx in enumerate(sorted_idx):
    print(f"[{i}] {texts[idx]}")
    print()

model_name = "intfloat/multilingual-e5-large-instruct"
# model_name = "intfloat/multilingual-e5-large"

hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},  # cuda, cpu
    encode_kwargs={"normalize_embeddings": True},
)

embedded_documents1 = hf_embeddings.embed_documents(texts)

print(f"Model: \t\t{model_name}")
print(f"Dimension: \t{len(embedded_documents1[0])}")

model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

embedded_documents = hf_embeddings.embed_documents(texts)

print(f"Model: \t\t{model_name}")
print(f"Dimension: \t{len(embedded_documents[0])}")