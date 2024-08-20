# File konfigurasi untuk mengelola API key sebagai variabel lingkungan
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Memuat informasi API key
load_dotenv()

# Membuat embedding menggunakan model "text-embedding-3-large" dari OpenAI.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

text = "Ini adalah kalimat contoh untuk menguji embedding."

# Membuat hasil kueri dengan meng-embedding teks.
query_result = embeddings.embed_query(text)

# Memilih 5 item pertama dari hasil kueri.
print(query_result[:5])

doc_result = embeddings.embed_documents(
    [text]
)  # Membuat vektor dokumen dengan meng-embedding teks.

# Memilih 5 item pertama dari elemen pertama hasil dokumen.
print(doc_result[0][:5])

# Mengembalikan panjang elemen pertama dari hasil dokumen.
print(len(doc_result[0]))


# Menginisialisasi objek yang membuat embedding 1024 dimensi menggunakan model "text-embedding-3-small" dari OpenAI.
embeddings_1024 = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)

# Membuat embedding dari teks yang diberikan dan mengembalikan panjang vektor embedding pertama.
print(len(embeddings_1024.embed_documents([text])[0]))


sentence1 = "Halo? Senang bertemu dengan Anda."
sentence2 = "Halo? Senang bertemu dengan Anda!"
sentence3 = "Halo? Senang bertemu denganmu"
sentence4 = "Hi, nice to meet you."
sentence5 = "I like to eat apples."

sentences = [sentence1, sentence2, sentence3, sentence4, sentence5]
embedded_sentences = embeddings_1024.embed_documents(sentences)

def similarity(a, b):
    return cosine_similarity([a], [b])[0][0]

for i, sentence in enumerate(embedded_sentences):
    for j, other_sentence in enumerate(embedded_sentences):
        if i < j:
            print(
                f"[Kesamaan {similarity(sentence, other_sentence):.4f}] {sentences[i]} \t <=====> \t {sentences[j]}"
            )