from dotenv import load_dotenv
import uuid
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import MultiVectorRetriever
from langchain_core.stores import InMemoryByteStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

loaders = [
    # Muat file.
    TextLoader("./data/ai-story.txt"),
    # Muat file.
    TextLoader("./data/appendix-keywords.txt"),
]

docs = []  # Menginisialisasi daftar dokumen kosong.
for loader in loaders:
    docs.extend(loader.load())  # Memuat dokumen dari setiap loader dan menambahkannya ke dalam daftar docs.

vectorstore = Chroma(
    collection_name="full_documents", embedding_function=OpenAIEmbeddings()
)
# Penyimpanan tingkat penyimpanan untuk dokumen induk
store = InMemoryByteStore()

id_key = "doc_id"
# Retriever (kosong saat dimulai)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

# Menghasilkan ID dokumen.
doc_ids = [str(uuid.uuid4()) for _ in docs]
# Memeriksa dua ID yang dihasilkan.
print(doc_ids)

# Membuat objek RecursiveCharacterTextSplitter.
parent_text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000)

# Splitter yang akan digunakan untuk membuat chunk yang lebih kecil
child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

parent_docs = []

for i, doc in enumerate(docs):
    _id = doc_ids[i]  # Mengambil ID dari dokumen saat ini.
    # Membagi dokumen saat ini menjadi dokumen-dokumen kecil.
    parent_doc = parent_text_splitter.split_documents([doc])
    for _doc in parent_doc:  # Mengulang setiap dokumen yang telah dibagi.
        # Menyimpan ID di metadata dokumen.
        _doc.metadata[id_key] = _id
    parent_docs.extend(parent_doc)  # Menambahkan dokumen yang telah dibagi ke dalam daftar.

child_docs = []  # Menginisialisasi daftar untuk menyimpan dokumen-dokumen kecil.
for i, doc in enumerate(docs):
    _id = doc_ids[i]  # Mengambil ID dari dokumen saat ini.
    # Membagi dokumen saat ini menjadi dokumen-dokumen kecil.
    child_doc = child_text_splitter.split_documents([doc])
    for _doc in child_doc:  # Mengulang setiap dokumen kecil yang telah dibagi.
        # Menyimpan ID di metadata dokumen kecil.
        _doc.metadata[id_key] = _id
    child_docs.extend(child_doc)  # Menambahkan dokumen kecil yang telah dibagi ke dalam daftar.

print(f"Jumlah parent_docs yang telah dibagi: {len(parent_docs)}")
print(f"Jumlah child_docs yang telah dibagi: {len(child_docs)}")

# Menambahkan dokumen ke repositori vektor.
retriever.vectorstore.add_documents(parent_docs)
retriever.vectorstore.add_documents(child_docs)

# Menyimpan ID dokumen dan dokumen ke dalam penyimpanan dokumen dengan memetakannya.
retriever.docstore.mset(list(zip(doc_ids, docs)))

# Lakukan pencarian kemiripan di vectorstore.
sim_search = retriever.vectorstore.similarity_search("Apa definisi dari Word2Vec?")
print(sim_search)

# Melakukan pencarian kesamaan menggunakan score_threshold.
sim_search_w_threshold = retriever.vectorstore.similarity_search_with_relevance_scores(
    "Apa definisi dari Word2Vec?", score_threshold=0.5, k=3
)
print()
print(sim_search_w_threshold)

relevant_doc = retriever.get_relevant_documents("Apa definisi dari Word2Vec?")
docs_len = len(relevant_doc)
print(docs_len)

# Periksa panjang dokumen yang dikembalikan oleh Retriever.
docs_content_len = len(retriever.get_relevant_documents("Definisi Word2Vec?")[0].page_content)
print(docs_content_len)

from langchain.retrievers.multi_vector import SearchType

# Mengatur jenis pencarian ke MMR (Maximal Marginal Relevance)
retriever.search_type = SearchType.mmr

# Cari dokumen terkait menggunakan query, dan kembalikan panjang konten halaman dari dokumen pertama
docs_content_len = len(retriever.get_relevant_documents("Definisi Word2Vec")[0].page_content)
print(docs_content_len)

import uuid

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


chain = (
    {"doc": lambda x: x.page_content}  # Fungsi untuk mengekstrak konten halaman dari data input
    # Membuat template prompt untuk merangkum dokumen
    | ChatPromptTemplate.from_template(
        "Ringkas dokumen berikut dalam bahasa Indonesia:\n\n{doc}"
    )
    # Menggunakan model ChatGPT dari OpenAI untuk membuat ringkasan (Jumlah percobaan ulang maksimal: 0)
    | ChatOpenAI(max_retries=0)
    | StrOutputParser()  # Mengubah hasil ringkasan menjadi string
)

# Memproses sekumpulan dokumen dengan konkurensi maksimum yang disetel ke 5
summaries = chain.batch(docs, {"max_concurrency": 5})

print("summaries\n")

# Mencetak ringkasan.
print(summaries[0])

print("summaries1\n")

print(summaries[1])

# Membuat penyimpanan vektor untuk menyimpan ringkasan informasi.
vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())
# Membuat penyimpanan untuk menyimpan dokumen induk.
store = InMemoryByteStore()
# Menetapkan nama kunci untuk menyimpan ID dokumen.
id_key = "doc_id"
# Menginisialisasi retriever (pada awalnya kosong).
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,  # Penyimpanan vektor
    byte_store=store,  # Penyimpanan byte
    id_key=id_key,  # Kunci ID dokumen
)
# Membuat ID dokumen.
doc_ids = [str(uuid.uuid4()) for _ in docs]

summary_docs = [
    # Membuat objek Document dengan konten halaman yang berisi ringkasan dan metadata yang berisi ID dokumen.
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(
        summaries
    )  # Mengulangi setiap ringkasan dan indeks dalam daftar summaries.
]

# Jumlah dokumen dalam intisari
docs_len = len(summary_docs)

print(docs_len)

retriever.vectorstore.add_documents(
    summary_docs
)  # Tambahkan dokumen yang telah diringkas ke tempat penyimpanan vektor.

# Memetakan ID dokumen ke dokumen dan menyimpannya di penyimpanan dokumen.
retriever.docstore.mset(list(zip(doc_ids, docs)))

# Membuat objek RecursiveCharacterTextSplitter.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

split_docs = []
split_docs_ids = []

for i, doc in enumerate(docs):
    _id = doc_ids[i]  # Mendapatkan ID dokumen saat ini.
    # Membagi dokumen saat ini menjadi sub-dokumen.
    split_doc = text_splitter.split_documents([doc])
    for _doc in split_doc:  # Mengulangi setiap sub-dokumen yang dibagi.
        # Menyimpan ID di metadata dokumen.
        _doc.metadata[id_key] = _id
        split_docs_ids.append(_id)
    split_docs.extend(split_doc)  # Menambahkan sub-dokumen yang telah dibagi ke dalam daftar.

print(f"Jumlah dokumen yang dipisahkan: {len(split_docs)}")

# Tambahkan dokumen ke penyimpanan vektor pencari.
added = retriever.vectorstore.add_documents(split_docs)
print("addedd")
print(added)

# Lakukan pencarian kemiripan.
result_docs = vectorstore.similarity_search("Apa definisi dari Word2Vec?")
print("result docs[0]")
print(result_docs[0])

# Mencari dan mengambil dokumen terkait.
retrieved_docs = retriever.get_relevant_documents("Apa definisi dari Word2Vec?")
relevant_doc_len = len(retrieved_docs)
print(relevant_doc_len)

functions = [
    {
        "name": "hypothetical_questions",  # Menentukan nama fungsi.
        "description": "Generate hypothetical questions",  # Menulis deskripsi fungsi.
        "parameters": {  # Mendefinisikan parameter fungsi.
            "type": "object",  # Menentukan tipe parameter sebagai objek.
            "properties": {  # Mendefinisikan properti objek.
                "questions": {  # Mendefinisikan properti 'questions'.
                    "type": "array",  # Menentukan tipe 'questions' sebagai array.
                    "items": {
                        "type": "string"
                    },  # Menentukan tipe elemen dalam array sebagai string.
                },
            },
            "required": ["questions"],  # Menentukan 'questions' sebagai parameter wajib.
        },
    }
]

from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser

chain = (
    {"doc": lambda x: x.page_content}
    # Meminta untuk menghasilkan tepat 3 pertanyaan hipotetis yang dapat dijawab dengan menggunakan dokumen di bawah ini. Angka ini dapat disesuaikan.
    | ChatPromptTemplate.from_template(
        "Buatlah daftar tepat 3 pertanyaan hipotetis yang dapat dijawab dengan menggunakan dokumen di bawah ini. Jawab dalam bahasa Indonesia:\n\n{doc}"
    )
    | ChatOpenAI(max_retries=0, model="gpt-4-turbo-preview").bind(
        functions=functions, function_call={"name": "hypothetical_questions"}
    )
    # Mengekstrak nilai yang terkait dengan kunci "questions" dari output.
    | JsonKeyOutputFunctionsParser(key_name="questions")
)

# Jalankan rantai untuk dokumen yang diberikan.
chain_res = chain.invoke(docs[0])
print(chain_res)

# Membuat pertanyaan hipotetis dalam batch untuk daftar dokumen. Konkurensi maksimum diatur ke 5.
hypothetical_questions = chain.batch(docs, {"max_concurrency": 5})

print("hypothetical questions")
print(hypothetical_questions[0])
print(hypothetical_questions[1])

# Penyimpanan vektor yang akan digunakan untuk mengindeks chunk anak
vectorstore = Chroma(
    collection_name="hypo-questions", embedding_function=OpenAIEmbeddings()
)
# Tingkatan penyimpanan untuk dokumen induk
store = InMemoryByteStore()
id_key = "doc_id"
# Pencari (awalnya kosong)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
doc_ids = [str(uuid.uuid4()) for _ in docs]  # Pembuatan ID dokumen

question_docs = []
# Melakukan iterasi melalui daftar hypothetical_questions sambil mengambil indeks dan daftar pertanyaan.
for i, question_list in enumerate(hypothetical_questions):
    question_docs.extend(  # Menambahkan objek Document ke dalam daftar question_docs.
        # Membuat objek Document untuk setiap pertanyaan dalam daftar pertanyaan, dan menyertakan ID dokumen terkait dalam metadata.
        [Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in question_list]
    )

retriever.vectorstore.add_documents(
    question_docs
)  # Tambahkan dokumen pertanyaan ke tempat penyimpanan vektor.
# Memetakan ID dokumen ke dokumen dan menyimpannya ke penyimpanan dokumen.
retriever.docstore.mset(list(zip(doc_ids, docs)))

# Cari repositori vektor untuk dokumen serupa.
result_docs = vectorstore.similarity_search("Apa definisi dari Word2Vec?")

print(result_docs)

# Tambahkan dokumen ke penyimpanan vektor pencari.
splitted_docs = retriever.vectorstore.add_documents(split_docs)
print(f"splitted = {splitted_docs}")

retrieved_docs = retriever.get_relevant_documents("Apa definisi dari Word2Vec?")
print(f"len {len(retrieved_docs)}")

print(f"content len {len(retrieved_docs[0].page_content)}")