from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage
class PromptLib():
    history = None
    systemPromptTemplae = []
    humanPromptTemplate = []
    examples = []
    def __init__(self,langchainLib):
        self.langchainLib = langchainLib
        self.history = []
    def get_prompt(self,systemTemplate:str,humanTemplate:str,examples:list[str]=[]):
        if self.examples:
            example_str =  "\nExamples:\n"
            for example in examples:
                example_str += example + "\n"
            systemTemplate = systemTemplate +example_str        
        self.systemPromptTemplae = [('system',systemTemplate)]
        self.humanPromptTemplate = [('human',humanTemplate)]
        self.examples = examples
        messages = self.systemPromptTemplae + self.history + self.humanPromptTemplate
        chatPromptTemplate = ChatPromptTemplate.from_messages(messages = messages)
        return chatPromptTemplate
    def update(self,humanMsg,aiMsg):
        self.history.append(HumanMessage(humanMsg))
        self.history.append(AIMessage(aiMsg))
        messages = self.systemPromptTemplae + self.history + self.humanPromptTemplate
        chatPromptTemplate = ChatPromptTemplate.from_messages(messages = messages)
        return chatPromptTemplate
    def reset(self):
        self.history = []
        messages = self.systemPromptTemplae + self.history + self.humanPromptTemplate
        chatPromptTemplate = ChatPromptTemplate.from_messages(messages = messages)
        return chatPromptTemplate
    def print_pretty(self,prompt:ChatPromptTemplate):
        input_variables = prompt.input_variables
        print(f"++++++++++{input_variables}+++++++++++++++")
        for idx,message in enumerate(prompt.messages):
            print(idx,message)