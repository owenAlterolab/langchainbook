from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub

load_dotenv()
repo_id = "google/flan-t5-xxl"

t5_model = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 512}
)

t5_model.invoke("Where is the capital of South Korea?")