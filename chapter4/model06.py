from dotenv import load_dotenv
from langchain_teddynote import logging
from huggingface_hub import login
from langchain.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()
logging.langsmith("CH04-Models")

# login()

template = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
{question}<|end|>
<|assistant|>"""

prompt = PromptTemplate.from_template(template)

# 사용할 모델의 저장소 ID를 설정합니다.
repo_id = "microsoft/Phi-3-mini-4k-instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,  # 모델 저장소 ID를 지정합니다.
    max_new_tokens=256,  # 생성할 최대 토큰 길이를 설정합니다.
    temperature=0.1,
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],  # 허깅페이스 토큰
)

# LLMChain을 초기화하고 프롬프트와 언어 모델을 전달합니다.
chain = prompt | llm | StrOutputParser()
# 질문을 전달하여 LLMChain을 실행하고 결과를 출력합니다.
response = chain.invoke({"question": "what is the capital of Indonesia?"})
print(response)