
from youht_app1.langchain.llm import LLMLib

class LangchainLib():
    llmLib = None
    def __init__(self):
        print("langchainLib init...")
        self.llmLib = LLMLib()