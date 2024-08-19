from dotenv import load_dotenv
from langchain_altero import logging
from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH04-Models")
llm = ChatOpenAI(model_name="gpt-4o")

# Gunakan callback untuk token.
with get_openai_callback() as cb:
    result = llm.invoke("Bakso berbentuk seperti apa?")
    print(cb)

# Callback untuk melacak penggunaan.
with get_openai_callback() as cb:
    result = llm.invoke("Bakso berbentuk seperti apa??")
    result = llm.invoke("Bakso berbentuk seperti apa??")
    print(f"Total token yang digunakan: \t\t{cb.total_tokens}")
    print(f"Token yang digunakan untuk prompt: \t{cb.prompt_tokens}")
    print(f"Token yang digunakan untuk jawaban: \t{cb.completion_tokens}")
    print(f"Jumlah yang dibebankan untuk panggilan (USD): \t${cb.total_cost}")