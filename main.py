import types
from typing import Optional
from langchain_core.documents import Document
from youht_app1.langchain import LangchainLib
from langchain_core.messages import AIMessage,AIMessageChunk
from langchain.pydantic_v1 import BaseModel,Field
from langchain_core.tools import tool
class MyOutput(BaseModel):
    name: Optional[str] = Field(description="姓名") 
    address: Optional[str] = Field(description="地址")
    age: Optional[int] = Field(description="年龄",gt=0,lt=100)

def add(a:float ,b :float):
    '''add two number'''
    return a+b
def multi(a:float ,b :float):
    '''multi two number'''
    return a*b
def repl(script:str):
    '''执行python表达式,比如3.5**2,或者time.now()。当不知道问题的答案，需要通过函数来获得答案时使用此函数
    由于使用exec执行script，所以需要在script最后一条语句通过RESULT= 来设置返回结果
    '''
    new_scripts = script.split('\n')
    if not new_scripts[-1].strip().upper().startswith("RESULT"):
        print("手动添加",new_scripts[-1].strip().upper())
        new_scripts[-1] = f"RESULT =  {new_scripts[-1]}"
    script = "\n".join(new_scripts)
    namespace = {}
    exec(script,globals(),namespace)
    result = namespace.get('RESULT')
    return result
@tool
def no_search_web(input:str):
    ''' 需要进行web查询是产生错误 '''
    return 'we can not search web'
@tool
def no_add(input:str):
    ''' 当需要进行加法运算时产生错误 '''
    return 'we can not use add '

def main():
    langchainLib = LangchainLib()
    llm = langchainLib.llmLib.get_llm("DEEPBRICKS_API_KEY","gpt-4o-mini","https://api.deepbricks.ai/v1")
    systemTemplate = f"所有的回答都用{{language}}，语气调皮一些，多加一下表情符号"
    humanTemplate = f"input:{{input}}"
    prompt = langchainLib.promptLib.get_prompt(systemTemplate,humanTemplate)
    language = "中文"
    #llm = llm.with_structured_output(MyOutput)
    #llm = llm.with_fallbacks([no_search_web,no_add])
    tools = [add,multi,repl]
    llm = llm.bind_tools(tools)
    while True:
        msg = input("User: ")
        if msg.lower() == '/q':
            break
        if msg.lower() == '/c':
            prompt = langchainLib.promptLib.reset()
            continue
        if msg.lower().startswith('/l'):
            print(f"之前我使用{language}")
            language = msg.split(' ')[1].strip()
            print(f"现在我将使用{language}回答")
            continue
        chain = prompt | llm
        try:
            chunks  =  chain.invoke(
                {"language":language,"input": msg}
            )
        except Exception as e:
            print(e)
            continue
        langchainLib.promptLib.print_pretty(prompt)
        print("AI:",end="")
        aiMsg = ""
        if isinstance(chunks,AIMessage): #是invoke调用,且没有使用output parser
            for chunk in chunks:
                if chunk[0]=="content":       #是普通输出
                    print(chunk[1],end="")
                    aiMsg += chunk[1]            
                elif chunk[0]=="tool_calls":  #是tool_calls
                    if chunk[1]:
                        print(f"    \ntool_calls:")
                        for tool_item in chunk[1]:
                            print(f"        name:{tool_item['name']},args:{tool_item['args']}")
                            try:
                                for tool in tools:
                                    if tool.__name__ == tool_item['name']:
                                        result = tool(**tool_item['args'])
                                        aiMsg += f"the result is : {result}\n"
                            except Exception as e:
                                aiMsg = f"error on execute function `{tool_item['name']}` with args {tool_item['args']}.\nThe error message is {str(e)}"
                        print(f"\n  {aiMsg}")
        elif isinstance(chunks, types.GeneratorType): #是stream调用
            for chunk in chunks:
                if chunk.content:
                    print(chunk.content,end='')
                    aiMsg += chunk.content
        else:     #指定了output的输出
            print(f"\n    chunks type={type(chunks)}")
            print(f"        chunks = {chunks}")
        print("\n")
        prompt = langchainLib.promptLib.update(msg,aiMsg)
    print("good bye!")        
    
if __name__ == '__main__':
    main()
    