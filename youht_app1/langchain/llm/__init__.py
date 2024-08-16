from langchain_openai import ChatOpenAI
import os

class LLMLib():
    def __init__(self):
        print("LLMLib init...")
    def get_llm(self,llm_key,llm_model,base_url):
        llm = ChatOpenAI(
            api_key = os.environ.get(llm_key),
            base_url= base_url,
            model = llm_model
        ) 
        return llm