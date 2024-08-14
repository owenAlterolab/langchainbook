import os
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["TRANSFORMERS_CACHE"] = "./cache/"
os.environ["HF_HOME"] = "./cache/"

model_id = "beomi/llama-2-ko-7b"  # 사용할 모델의 ID를 지정합니다.
tokenizer = AutoTokenizer.from_pretrained(
    model_id
)  # 지정된 모델의 토크나이저를 로드합니다.
model = AutoModelForCausalLM.from_pretrained(model_id)  # 지정된 모델을 로드합니다.
# 텍스트 생성 파이프라인을 생성하고, 최대 생성할 새로운 토큰 수를 10으로 설정합니다.
pipe = pipeline("text-generation", model=model,
                tokenizer=tokenizer, max_new_tokens=512)
# HuggingFacePipeline 객체를 생성하고, 생성된 파이프라인을 전달합니다.
hf = HuggingFacePipeline(pipeline=pipe)

template = """Answer the following question in Bahasa Indonesia.
#Question: 
{question}

#Answer: """  # 질문과 답변 형식을 정의하는 템플릿
prompt = PromptTemplate.from_template(template)  # 템플릿을 사용하여 프롬프트 객체 생성

# 프롬프트와 언어 모델을 연결하여 체인 생성
chain = prompt | hf | StrOutputParser()

question = "Buah jeruk warnanya apa ?"  # 질문 정의

print(
    chain.invoke({"question": question})
)  # 체인을 호출하여 질문에 대한 답변 생성 및 출력