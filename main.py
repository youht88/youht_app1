from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate,StringPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,ChatPromptTemplate
from youht_app1.langchain import LangchainLib

def main():
    langchainLib = LangchainLib()
    llmLib = langchainLib.llmLib
    llm = llmLib.get_llm("DEEPBRICKS_API_KEY","gpt-4o-mini","https://api.deepbricks.ai/v1")
    messages = []
    systemPrompt = PromptTemplate(template = "所有的回答都用{language}，语气调皮一些，多加一下表情符号")
    messages.append(SystemMessagePromptTemplate(prompt = systemPrompt))
    while True:
        msg = input("User: ")
        if msg.lower() == '/q':
            break
        if msg.lower() == '/c':
            messages = messages[:1]
        #messages.append(HumanMessagePromptTemplate(prompt=PromptTemplate(msg)))
        humanPrompt = PromptTemplate(template = "input:{input}")
        messages.append(HumanMessagePromptTemplate(prompt = humanPrompt))
        #messages.append(("human",msg))
        #messages.append(HumanMessage(msg))
        print("???",messages)
        chatPromptTemplate = ChatPromptTemplate.from_messages(messages = messages)
        print("????",chatPromptTemplate)
        new_chatPromptTemplate = chatPromptTemplate.partial(language = "法文") 
        print("!!!",new_chatPromptTemplate,type(new_chatPromptTemplate))
        chunks  =  new_chatPromptTemplate | llm.invoke(
            input = msg
        )
        print("===>",chunks)
        print("AI:",end="")
        aiMsg = ""
        for chunk in chunks:
            print(chunk.content,end="")
            aiMsg += chunk.content
        print("\n")
        messages.append(AIMessage(aiMsg))
    print("good bye!")        
    
if __name__ == '__main__':
    main()
    