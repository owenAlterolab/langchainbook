from langchain.output_parsers import OutputFixingParser
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()


class Actor(BaseModel):
    name: str = Field(description="name of an actor")
    film_names: List[str] = Field(
        description="list of names of films they starred in")


actor_query = "Generate the filmography for a random actor."

parser = PydanticOutputParser(pydantic_object=Actor)
# 잘못된 형식을 일부러 입력
misformatted = '{"name": "Tom Hanks", "film_names": ["Forrest Gump"]}'

# 잘못된 형식으로 입력된 데이터를 파싱하려고 시도
# tst = parser.parse(misformatted)
# 오류 출력

new_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())
# 잘못된 형식의 출력

print(misformatted)

actor = new_parser.parse(misformatted)

print(actor)