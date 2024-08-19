from dotenv import load_dotenv
from langchain_altero import logging
from langchain_altero.models import MultiModal
from langchain_altero.messages import stream_response
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

# TODO: NEED GOOGLE GEMINI AI API KEY

load_dotenv()
logging.langsmith("CH04-Models")

# Inisialisasi model bahasa ChatGoogleGenerativeAI.
# llm = ChatGoogleGenerativeAI(model = "gemini-1.5-pro-latest")

# # Berikan perintah untuk menghasilkan hasil.
# answer = llm.stream("Ceritakan sedikit tentang Indonesia")

# # Keluarkan hasilnya.
# stream_response(answer)

# Inisialisasi model bahasa ChatGoogleGenerativeAI.
# model = ChatGoogleGenerativeAI(
#     model = "gemini-1.5-flash-latest", # Tentukan model yang akan digunakan.
# )

# # Membuat prompt.
# prompt = PromptTemplate.from_template(
#     "Jawab pertanyaan ya/tidak. Apakah {pertanyaan} adalah buah?"
# )

# # Membuat sebuah rantai.
# chain = prompt | model

# # Keluarkan hasilnya.
# stream_response(chain.stream({"pertanyaan": "apel"}))

# llm = ChatGoogleGenerativeAI(
#     # 사용할 모델을 "gemini-pro"로 지정합니다.
#     model="gemini-1.5-pro-latest",
#     safety_settings={
#         # 위험한 콘텐츠에 대한 차단 임계값을 설정합니다.
#         # 이 경우 위험한 콘텐츠를 차단하지 않도록 설정되어 있습니다. (그럼에도 기본적인 차단이 있을 수 있습니다.)
#         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     },
# )

llm = ChatGoogleGenerativeAI(
    # 사용할 모델을 "gemini-pro"로 지정합니다.
    model="gemini-1.5-pro-latest",
)

# results = llm.batch(
#     [
#         "Apa ibu kota Indonesia?",
#         "Sebutkan 5 tempat wisata teratas di Indonesia",
#     ]
# )

# for res in results:
#     # 각 결과의 내용을 출력합니다.
#     print(res.content)

# membuat sebuah objek
gemini = ChatGoogleGenerativeAI(model = "gemini-1.5-pro-latest")

system_prompt = (
    "Anda adalah seorang penyair, tugas Anda adalah menulis puisi dengan gambar yang diberikan."
)

user_prompt = "Tolong tulislah sebuah puisi tentang gambar berikut ini."

# Membuat objek multimodal
multimodal_gemini = MultiModal(
    llm, system_prompt = system_prompt, user_prompt = user_prompt
)

# need full path to images
IMAGE_URL = "/Users/mini16gboffice1/StudioProjects/owen/python/langchainbook/chapter4/images/beach.jpeg"

# 이미지 파일로 부터 질의
answer = multimodal_gemini.stream(IMAGE_URL)

# 스트리밍 방식으로 각 토큰을 출력합니다. (실시간 출력)
stream_response(answer)