from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_teddynote.messages import stream_response
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_upstage import ChatUpstage

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH04-Models")

gpt = ChatOpenAI(
    temperature=0.1,  # 창의성 (0.0 ~ 2.0)
    model_name="gpt-4o",  # 모델명
)

question = "apa ibukota negara indonesia?"

# print(f"[jawaban]: {gpt.invoke(question)}")
answer = gpt.stream("apa ibukota negara indonesia?")

# 답변 출력
stream_response(answer)

# TODO: NEED anthropic API KEY
# # ChatAnthropic 객체를 생성합니다.
# anthropic = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")

# # 스트리밍 출력을 위하여 invoke() 대신 stream()을 사용합니다.
# answer = anthropic.stream("사랑이 뭔가요?")

# # 답변 출력
# stream_response(answer)


# TODO: NEED COHERE API KEY
# cohere = ChatCohere(temperature=0)

# # 스트리밍 출력을 위하여 invoke() 대신 stream()을 사용합니다.
# answer = cohere.stream("사랑이 뭔가요?")

# # 답변 출력
# stream_response(answer)

# TODO: NEED UPSTAGE_API_KEY
# # ChatUpstage 객체를 생성합니다.
# upstage = ChatUpstage()

# # 스트리밍 출력을 위하여 invoke() 대신 stream()을 사용합니다.
# answer = upstage.stream("사랑이 뭔가요?")

# # 답변 출력
# stream_response(answer)

# ! MODEL XIONIC DOES NOT EXIST
# xionic = ChatOpenAI(
#     model_name="xionic-1-72b-20240610",
#     base_url="https://sionic.chat/v1/",
#     api_key="934c4bbc-c384-4bea-af82-1450d7f8128d",
# )

# # 스트리밍 출력을 위하여 invoke() 대신 stream()을 사용합니다.
# answer = xionic.stream("사랑이 뭔가요?")

# # 답변 출력
# stream_response(answer)