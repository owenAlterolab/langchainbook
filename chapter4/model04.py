from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH04-Models")
llm = ChatOpenAI(model_name="gpt-4o")

# Gunakan callback untuk token.
with get_openai_callback() as cb:
    hasil = llm.invoke("Apa ibu kota Indonesia?")
    hasil = llm.invoke("Apa ibu kota Indonesia?")
    print(cb)