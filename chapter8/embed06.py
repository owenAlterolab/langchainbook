from langchain_community.embeddings import GPT4AllEmbeddings

gpt4all_embd = GPT4AllEmbeddings()  # Membuat objek embedding GPT4All.

text = (
    "Ini adalah kalimat contoh untuk menguji embedding."  # Mendefinisikan teks dokumen untuk pengujian.
)

query_result = gpt4all_embd.embed_query(
    text
)  # Membuat embedding kueri untuk teks yang diberikan.

# Periksa ukuran dimensi yang disematkan.
print(len(query_result))

# Meng-embedding teks yang diberikan untuk menghasilkan vektor dokumen.
doc_result = gpt4all_embd.embed_documents([text])

# Memeriksa ukuran dimensi yang di-embedding.
print(len(doc_result[0]))