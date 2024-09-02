from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import GPT4All
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import StreamingStdOutCallbackHandler

local_path = "C:/Users/owenl/AppData/Local/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf"  # 원하는 로컬 파일 경로로 대체하세요.

# 프롬프트
prompt = ChatPromptTemplate.from_template(
    """<s>Percakapan antara seorang pengguna yang penasaran dan asisten kecerdasan buatan. Asisten memberikan jawaban yang membantu, terperinci, dan sopan atas pertanyaan-pertanyaan pengguna.</s>
<s>Human: {question}</s>
<s>Assistant:
"""
)

# GPT4All 언어 모델 초기화
# model는 GPT4All 모델 파일의 경로를 지정
llm = GPT4All(
    model=local_path,
    backend="gpu",  # GPU 설정
    streaming=True,  # 스트리밍 설정
    callbacks=[StreamingStdOutCallbackHandler()],  # 콜백 설정
)

# 체인 생성
chain = prompt | llm | StrOutputParser()

# 질의 실행
response = chain.invoke({"question": "Apa nama ibukota Indoesia?"})