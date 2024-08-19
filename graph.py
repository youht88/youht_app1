from youht_app1.langchain import LangchainLib
from langgraph.graph import START,END,StateGraph
from langgraph.graph.message import MessagesState
from langchain.prompts import StringPromptTemplate,PromptTemplate,ChatPromptTemplate
from langchain_experimental.tools import PythonAstREPLTool
from langgraph.prebuilt import ToolNode
from typing import Literal
from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel,Field
from langchain_core.messages import ToolMessage

langchainLib = LangchainLib()
llm = langchainLib.llmLib.get_llm("DEEPBRICKS_API_KEY","gpt-4o-mini","https://api.deepbricks.ai/v1")
    
class PythonREPLArgSchema(BaseModel):
    ''' input str for python repl tool'''
    command: str = Field(description="the command to execute use python repl")
repl = PythonAstREPLTool()
repl_tool = Tool(
            name="repl_tool",
            description="""
The tool is a Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.""",
            args_schema = PythonREPLArgSchema ,
            func=repl.run
        )

tools = [repl_tool]
tools_node = ToolNode(tools=tools) 
llm_with_tools = llm.bind_tools(tools)

def node1(state:MessagesState):
    messages = state["messages"]
    message = messages[-1]
    llm_prompt = PromptTemplate(template = f"把这个句子翻译成英文:{{input}}") 
    chain = llm_prompt | llm 
    aiMessage = chain.invoke({"input":message.content})
    return {"messages":[aiMessage]}

def node2(state:MessagesState):
    messages = state["messages"]
    message = messages[-1]
    if isinstance(message,ToolMessage):
        print("!"*80,messages[-2:])
        #messages.pop()
        #messages.append = ('ai',f'The result is :{message.content}')
    aiMessage = llm_with_tools.invoke(messages[-2:])
    print("@"*80)
    return {"messages":[aiMessage]}

def should_use_tools(state:MessagesState)->Literal['tools',END]:
    messages = state["messages"]
    message = messages[-1]
    print("should use tools:",message)
    if message.tool_calls and len(message.tool_calls)>0:
        return "tools"
    else:
        return END
    
def main():
    workflow = StateGraph(MessagesState)
    workflow.add_node("node1",node1)
    workflow.add_node("node2",node2)
    workflow.add_node("tools",tools_node)
    workflow.add_edge(START,"node1")
    workflow.add_edge("node1","node2")
    workflow.add_conditional_edges("node2",should_use_tools)
    workflow.add_edge("tools","node2")
    graph = workflow.compile()

    
    res = graph.stream({"messages":[('user','3.4的4.3次方是多少?')]})
    for chunk in res:
        print(chunk)
    
    # try:
    #     res = graph.invoke({"messages":[('user','3.4的4.3次方是多少?')]})
    #     print(res)
    # except Exception as e:
    #     print("*"*50,"ERROR")
    #     print(e)

if __name__ == "__main__":
    main()