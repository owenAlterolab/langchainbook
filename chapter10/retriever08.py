# Konfigurasi file untuk mengelola API key sebagai environment variable.
from dotenv import load_dotenv

# Muat informasi API key
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI

metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]

# Penjelasan singkat tentang isi dokumen
document_content_description = "Brief summary of a movie"

# Tentukan LLM
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

# Buat SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True
)

# Tentukan hanya filter yang Anda inginkan untuk melihat film dengan peringkat 8,5 atau lebih tinggi.
what_movie = retriever.invoke("I want to watch a movie rated higher than 8.5")
print(what_movie)

# Tanyakan apakah Greta Gerwig pernah menyutradarai film tentang wanita.
has_greta = retriever.invoke("Has Greta Gerwig directed any movies about women")
print(has_greta)

# Tentukan filter gabungan untuk mencari film fiksi ilmiah dengan peringkat 8,5 atau lebih tinggi.
whats_a_movie = retriever.invoke("What's a highly rated (above 8.5) science fiction film?")
print(whats_a_movie)

broken = retriever.invoke(
    # Cari film tentang mainan yang dibuat setelah tahun 1990 tetapi sebelum tahun 2005, tentukan filter gabungan dengan kueri bahwa film animasi lebih disukai.
    "What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated"
)
print(broken)

retriever = SelfQueryRetriever.from_llm(
    llm,  # Menentukan model bahasa (Language Model).
    vectorstore,  # Menentukan penyimpanan vektor (Vector Store).
    document_content_description,  # Menentukan deskripsi konten dokumen.
    metadata_field_info,  # Menentukan informasi bidang metadata.
    enable_limit=True,  # Mengaktifkan fitur pembatasan hasil pencarian.
    search_kwargs={"k": 2},  # Menetapkan nilai k menjadi 2 untuk membatasi hasil pencarian menjadi 2.
)

# Mengajukan pertanyaan tentang dua film yang berkaitan dengan dinosaurus.
two_dino = retriever.invoke("Apa saja film tentang dinosaurus?")
print(two_dino)

retriever = SelfQueryRetriever.from_llm(
    llm,  # Menentukan model bahasa (Language Model).
    vectorstore,  # Menentukan penyimpanan vektor (Vector Store).
    document_content_description,  # Menentukan deskripsi konten dokumen.
    metadata_field_info,  # Menentukan informasi bidang metadata.
    enable_limit=True,  # Mengaktifkan fitur pembatasan hasil pencarian.
)

# Menanyakan tentang dua film yang berkaitan dengan dinosaurus.
two_dino = retriever.invoke("Apa saja dua film tentang dinosaurus?")
print(two_dino)

one_dino = retriever.invoke("What are one movies about dinosaurs")
print(one_dino)

from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

# Ambil prompt generator query menggunakan deskripsi konten dokumen dan informasi bidang metadata.
prompt = get_query_constructor_prompt(
    document_content_description,
    metadata_field_info,
)

# Buat parser output query terstruktur dari komponen.
output_parser = StructuredQueryOutputParser.from_components()

# Hubungkan prompt, model bahasa, dan parser output untuk membuat query generator.
query_constructor = prompt | llm | output_parser

# mencetak hasil pemformatan string prompt dengan parameter kueri yang disetel ke “pertanyaan tiruan”.
# print(prompt.format(query="dummy question"))

query_invocation = query_constructor.invoke(
    {
        # Panggil generator kueri untuk menghasilkan kueri untuk pertanyaan yang diberikan.
        "query": "What are some sci-fi movies from the 90's directed by Luc Besson about taxi drivers"
    }
)
print(query_invocation)

from langchain_community.query_constructors.chroma import ChromaTranslator

retriever = SelfQueryRetriever(
    query_constructor=query_constructor, # Query generator yang dibuat sebelumnya
    vectorstore=vectorstore, # Tentukan penyimpanan vektor
    structured_query_translator=ChromaTranslator(), # Penerjemah query terstruktur
)

invoked = retriever.invoke(
    # Cari film tentang mainan yang dibuat setelah tahun 1990 tetapi sebelum tahun 2005, dengan preferensi pada film animasi.
    "What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated"
)

print(invoked)