import asyncio
from youht_app1.langchain import LangchainLib
from langgraph.graph import START,END,StateGraph
from langgraph.graph.message import MessagesState
from langchain.prompts import StringPromptTemplate,PromptTemplate,ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools import DuckDuckGoSearchRun,QuerySQLDataBaseTool
from langgraph.prebuilt import ToolNode
from typing import Literal
from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel,Field
from langchain_core.messages import ToolMessage,HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_community.utilities.sql_database import SQLDatabase

langchainLib = LangchainLib()
llm = langchainLib.llmLib.get_llm("DEEPBRICKS_API_KEY","gpt-4o-mini","https://api.deepbricks.ai/v1")
#llm = langchainLib.llmLib.get_llm("GROQ_API_KEY","llama-3.1-70b-versatile","https://api.groq.com/openai/v1")
#llm = langchainLib.llmLib.get_llm("GROQ_API_KEY","llama3-groq-70b-8192-tool-use-preview","https://api.groq.com/openai/v1")
    

repl = PythonREPLTool()
duckduckgo = DuckDuckGoSearchRun()
db = SQLDatabase.from_uri("sqlite:///abc.sqlite")
db_tool = QuerySQLDataBaseTool(db=db)


def add(x:float ,y:float):
    '''
      add two args
    '''
    return x+y

tools = [db_tool]
tools_node = ToolNode(tools=tools) 
llm_with_tools = llm.bind_tools(tools,strict=True)

checkpointer = MemorySaver()

def to_en(state:MessagesState):
    print("*"*50,"enter to_en")
    messages = state["messages"]
    message = messages[-1]
    llm_prompt = PromptTemplate(template = f"translate the sentence to english:{{input}}") 
    chain = llm_prompt | llm 
    aiMessage = chain.invoke({"input":message.content})
    return {"messages":[aiMessage]}

def node2(state:MessagesState):
    print("*"*50,"enter node2")
    messages = state["messages"]
    message = messages[-1]
    if isinstance(message,ToolMessage):
        print("!"*80)
        #message_pre = messages[-2]
        #message_pre.content = "I know the answer now"
    aiMessage = llm_with_tools.invoke(messages)
    return {"messages":[aiMessage]}

def to_cn(state:MessagesState):
    print("*"*50,"enter to_cn")
    messages = state["messages"]
    message = messages[-1]
    llm_prompt = PromptTemplate(template = f"把这个句子翻译成中文:{{input}}") 
    chain = llm_prompt | llm 
    aiMessage = chain.invoke({"input":message.content})
    return {"messages":[aiMessage]}

def should_use_tools(state:MessagesState)->Literal['tools','to_cn']:
    print("*"*50,"enter edge should use tools")
    messages = state["messages"]
    message = messages[-1]
    #print("should use tools:",message)
    if message.tool_calls and len(message.tool_calls)>0:
        return "tools"
    else:
        return "to_cn"

async def main():
    workflow = StateGraph(MessagesState)
    workflow.add_node("to_en",to_en)
    workflow.add_node("node2",node2)
    workflow.add_node("to_cn",to_cn)
    workflow.add_node("tools",tools_node)
    workflow.add_edge(START,"to_en")
    workflow.add_edge("to_cn",END)
    workflow.add_edge("to_en","node2")
    workflow.add_conditional_edges("node2",should_use_tools)
    workflow.add_edge("tools","to_cn")
    graph = workflow.compile(checkpointer=checkpointer)

    ############ astream
    #res = graph.stream({"messages":[('user','3.4的4.3次方是多少?')]},config={"configurable": {"thread_id": 42}})
    #res = graph.stream({"messages":[('user','what is the weather in sf')]},config={"configurable": {"thread_id": 42}})
    res = graph.astream({"messages":[('user','汉城奥运会上中国和美国各获得多少金牌?')]},config={"configurable": {"thread_id": 42}})
    async for chunk in res:
        for key,value in chunk.items():
            print("====>", key,"<=====")
            print(value["messages"][-1].content)
    
    #########  astream_log
    # res = graph.astream_log({"messages":[('user','汉城奥运会上中国和美国各获得多少金牌?')]},
    #                         config={"configurable": {"thread_id": 42}},
    #                         include_types=["llm"])
    # async for chunk in res:
    #     for op in chunk.ops:
    #         print(op["op"],op["path"])
    #         if op["path"] == "/streamed_output/-":
    #             print(op["value"])
    #         elif op["path"].startswith('/logs/') and op["path"].endswith("/streamed_output/-"):
    #             #if (op["value"].content):
    #             print(op["value"])
    
    ##########   astream_events
    # res = graph.astream_events({"messages":[('user','汉城奥运会上中国和美国各获得多少金牌?')]},
    #                         config={"configurable": {"thread_id": 42}},
    #                         version="v1")
    # async for event in res:
    #     #print(event["event"])
    #     #包括：on_chain_start,on_chain_stream,on_chain_end,on_tool_start,on_tool_end,
    #     # on_chat_model_start,on_chat_model_stream,on_chat_model_end
    #     kind = event["event"]
    #     if kind == "on_chat_model_stream":
    #         content = event["data"]["chunk"].content
    #         if content:
    #             print(content,end="|")
    #     elif kind == "on_chain_stream":
    #         content = event["data"]["chunk"]
    #         if content:
    #             print("on_chain_stream-->",content)
    #     elif kind == "on_tool_start":
    #         print("-"*20)
    #         print(f"String tool: {event['name']} with inputs: {event['data'].get('input')}")
    #     elif kind == "on_tool_end":
    #         print(f"Done tool: {event['name']}")
    #         print(f"tool output was: {event['data'].get('output')}")
    #         print("-"*20)


if __name__ == "__main__":
    pass
    #asyncio.run(main())
