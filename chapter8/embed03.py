from dotenv import load_dotenv
import os
import warnings
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings
import numpy as np
import time


load_dotenv()
warnings.filterwarnings("ignore")

texts = [
    "Headphone Bluetooth Nirkabel dengan Pembatalan Kebisingan.",
    "Speaker Bluetooth Tahan Air Portabel dengan Baterai 20 jam.",
    "Kursi Kantor Ergonomis dengan Dukungan Lumbar dan Ketinggian yang Dapat Disesuaikan.",
    "TV Smart 4K Ultra HD dengan Layanan Streaming dan Kontrol Suara.",
    "Meja Berdiri Elektrik dengan Tinggi yang Dapat Disetel Memori."
]

model_name = "intfloat/multilingual-e5-large-instruct"

hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=model_name,
    task="feature-extraction",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)

# Mulai waktu
start_time = time.time()

embedded_documents = hf_embeddings.embed_documents(texts)

# Akhiri waktu
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.4f} seconds")

print("[HuggingFace Endpoint Embedding]")
print(f"Model: \t\t{model_name}")
print(f"Dimension: \t{len(embedded_documents[0])}")

embedded_query = hf_embeddings.embed_query("Saya membutuhkan produk untuk ruangan kantor saya")

res1 = np.array(embedded_query) @ np.array(embedded_documents).T
print("np.array(embedded_query) @ np.array(embedded_documents).T")
print(res1)

sorted_idx = (np.array(embedded_query) @ np.array(embedded_documents).T).argsort()[::-1]
print("sorted_idx")
print(sorted_idx)

print("[Query] Saya membutuhkan produk untuk ruangan kantor saya\n====================================")
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

start_time = time.time()

embedded_documents1 = hf_embeddings.embed_documents(texts)

end_time = time.time()
print(f"Time taken: {elapsed_time:.4f} seconds")


print(f"Model: \t\t{model_name}")
print(f"Dimension: \t{len(embedded_documents1[0])}")

model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

start_time = time.time()

embedded_documents = hf_embeddings.embed_documents(texts)

end_time = time.time()
print(f"Time taken: {elapsed_time:.4f} seconds")

print(f"Model: \t\t{model_name}")
print(f"Dimension: \t{len(embedded_documents[0])}")

from FlagEmbedding import BGEM3FlagModel

model_name = "BAAI/bge-m3"
bge_embeddings = BGEM3FlagModel(
    model_name, use_fp16=True
)  # Mengatur use_fp16 ke True akan mempercepat proses perhitungan dengan sedikit penurunan kinerja.

bge_embedded = bge_embeddings.encode(
    texts,
    batch_size=12,
    max_length=8192,  # Jika panjang sebesar ini tidak diperlukan, Anda dapat mengatur nilai yang lebih kecil untuk mempercepat proses encoding.
)["dense_vecs"]

print("bge_embedded.shape")
print(bge_embedded.shape)

print(f"Model: \t\t{model_name}")
print(f"Dimension: \t{len(embedded_documents[0])}")

bge_encoded = bge_embeddings.encode(texts, return_dense=True)
print("bge_encoded['dense_vecs'].shape")
print(bge_encoded["dense_vecs"].shape)


bge_flagmodel = BGEM3FlagModel(
    "BAAI/bge-m3", use_fp16=True
)  # Mengatur use_fp16 ke True akan mempercepat proses perhitungan dengan sedikit penurunan kinerja.
bge_encoded = bge_flagmodel.encode(texts, return_sparse=True)

lexical_scores1 = bge_flagmodel.compute_lexical_matching_score(
    bge_encoded["lexical_weights"][0], bge_encoded["lexical_weights"][0]
)
lexical_scores2 = bge_flagmodel.compute_lexical_matching_score(
    bge_encoded["lexical_weights"][0], bge_encoded["lexical_weights"][1]
)
# 0 <-> 0
print(lexical_scores1)
# 0 <-> 1
print(lexical_scores2)


bge_flagmodel = BGEM3FlagModel(
    "BAAI/bge-m3", use_fp16=True
)  # Mengatur use_fp16 ke True akan mempercepat proses perhitungan dengan sedikit penurunan kinerja.
bge_encoded = bge_flagmodel.encode(texts, return_colbert_vecs=True)

colbert_scores1 = bge_flagmodel.colbert_score(
    bge_encoded["colbert_vecs"][0], bge_encoded["colbert_vecs"][0]
)
colbert_scores2 = bge_flagmodel.colbert_score(
    bge_encoded["colbert_vecs"][0], bge_encoded["colbert_vecs"][1]
)
# 0 <-> 0
print(colbert_scores1)
# 0 <-> 1
print(colbert_scores2)