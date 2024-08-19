
from youht_app1.langchain.llm import LLMLib
from youht_app1.langchain.prompts import PromptLib

class LangchainLib():
    llmLib = None
    def __init__(self):
        print("langchainLib init...")
        self.llmLib = LLMLib(self)
        self.promptLib = PromptLib(self)