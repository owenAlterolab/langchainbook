from langchain_teddynote import logging
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_teddynote.messages import stream_response
import asyncio
import base64
from io import BytesIO

from IPython.display import HTML, display
from PIL import Image
from langchain_core.messages import HumanMessage

logging.langsmith("CH04-Models")

# Memuat model Ollama.
llm = ChatOllama(model = "llama3.1:8b")

# Buat sebuah prompt
prompt = ChatPromptTemplate.from_template("Ceritakan sedikit tentang {topik}.")

# membuat rantai
chain = prompt | llm | StrOutputParser()

# Untuk ringkasnya, responsnya adalah keluaran ke terminal.
jawaban = chain.stream({"topik": "deep learning"})

# keluaran streaming
#stream_response(jawaban)

# async def test():
#     async for chunks in chain.astream(
#         {"topik": "Google"}
#     ):  # Menjalankan chain secara asinkron dan mengembalikan hasil dalam bentuk chunk.
#         print(chunks, end="", flush=True)  # Mencetak setiap chunk.

# asyncio.run(test())

llm = ChatOllama(
    model="llama3.1:8b",  # Menentukan model bahasa yang akan digunakan.
    format="json",  # Mengatur format input/output sebagai JSON.
    temperature=0,
)

# Membuat prompt yang meminta jawaban dalam format JSON
prompt = "Berikan 10 tempat wisata di Jawa timur. kunci: `places`. respons dalam format JSON."

# Memanggil chain
response = llm.invoke(prompt)
# print(response.content)  # Mencetak respons yang dihasilkan.


def convert_to_base64(pil_image):
    """
    Mengonversi gambar PIL menjadi string yang dienkode dalam Base64.

    :param pil_image: Gambar PIL
    :return: String Base64 yang terkode
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # Anda dapat mengubah format jika diperlukan.
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def plt_img_base64(img_base64):
    """
    Menampilkan gambar dari string yang dienkode dalam Base64.

    :param img_base64: String Base64
    """
    # Membuat tag HTML img dengan menggunakan string Base64 sebagai sumbernya
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # Merender HTML untuk menampilkan gambar
    display(HTML(image_html))


def prompt_func(data):  # Mendefinisikan fungsi prompt.
    text = data["text"]  # Mengambil teks dari data.
    image = data["image"]  # Mengambil gambar dari data.

    image_part = {  # Mendefinisikan bagian gambar.
        "type": "image_url",  # Menentukan tipe URL gambar.
        "image_url": f"data:image/jpeg;base64,{image}",  # Membuat URL gambar.
    }

    content_parts = []  # Menginisialisasi daftar untuk menyimpan bagian konten.

    text_part = {"type": "text", "text": text}  # Mendefinisikan bagian teks.

    content_parts.append(image_part)  # Menambahkan bagian gambar ke bagian konten.
    content_parts.append(text_part)  # Menambahkan bagian teks ke bagian konten.

    return [HumanMessage(content=content_parts)]  # Mengembalikan objek HumanMessage.


file_path = "/Users/mini16gboffice1/StudioProjects/owen/python/langchainbook/chapter4/images/beach.jpeg"  # Jalur file
pil_image = Image.open(file_path)

image_b64 = convert_to_base64(pil_image)  # Mengonversi gambar ke Base64

plt_img_base64(image_b64)  # Menampilkan gambar yang dienkode dalam Base64

# Memuat model bahasa multimodal ChatOllama.
llm = ChatOllama(model="llava:7b", temperature=0)

# Membuat chain dengan menghubungkan fungsi prompt, model bahasa, dan parser output.
chain = prompt_func | llm | StrOutputParser()

query_chain = chain.invoke(  # Memanggil chain untuk menjalankan query.
    # Mengirimkan teks dan gambar.
    {"text": "Deskripsikan gambar dalam poin-poin", "image": image_b64}
)

print(query_chain)  # Mencetak hasil query.
