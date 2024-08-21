import asyncio
from youht_app1.langchain import LangchainLib
from langgraph.graph import START,END,StateGraph
from langgraph.graph.message import MessagesState
from langchain_core.messages import AIMessage,RemoveMessage,SystemMessage
from langchain.prompts import StringPromptTemplate,PromptTemplate,ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools import DuckDuckGoSearchRun,QuerySQLDataBaseTool,InfoSQLDatabaseTool,ListSQLDatabaseTool
from langgraph.prebuilt import ToolNode
from typing import Literal
from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel,Field
from langchain_core.messages import ToolMessage,HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.runnables import RunnablePassthrough

langchainLib = LangchainLib()
#llm = langchainLib.llmLib.get_llm("DEEPBRICKS_API_KEY","gpt-4o-mini","https://api.deepbricks.ai/v1")
llm = langchainLib.llmLib.get_llm("GROQ_API_KEY","llama-3.1-70b-versatile","https://api.groq.com/openai/v1")
#llm = langchainLib.llmLib.get_llm("GROQ_API_KEY","llama3-groq-70b-8192-tool-use-preview","https://api.groq.com/openai/v1")
checkpointer = MemorySaver()
    

repl_tool = PythonREPLTool()
duckduckgo_tool = DuckDuckGoSearchRun()
db = SQLDatabase.from_uri("sqlite:///abc.sqlite")
db_query = QuerySQLDataBaseTool(db=db)
db_info = InfoSQLDatabaseTool(db=db)
db_tables = ListSQLDatabaseTool(db=db)
tools = [db_query]
tools_node = ToolNode(tools=tools) 
llm_with_tools = llm.bind_tools(tools,strict=True)

def node1(state:MessagesState):
    messages = state["messages"]
    message = messages[-1]
    # if isinstance(message,ToolMessage):
    #     remove_messages = [RemoveMessage(id = messages[-1].id),RemoveMessage(id = messages[-1].id)]
    #     add_messages = [("ai",f"I got the answer is :{message.content}"),
    #                  ("user","请用中文回答")
    #                  ]       
    #     res = llm_with_tools.invoke(messages[:-2]+add_messages)
    #     return {"messages":remove_messages+add_messages+[res]}
    # else:
    #     res = llm_with_tools.invoke(messages)
    #     return {"messages":[res]}
    chatPrompt = ChatPromptTemplate.from_messages(
        messages = [('system','必要时使用给定的数据库工具,\n涉及的表名:{tables}'),
                    *messages
                   ]).partial(tables = db.get_context())
    
    chain = chatPrompt | llm_with_tools
    res = chain.invoke({"a":123})
    return {"messages":[res]}

def should_continue(state:MessagesState):
    messages = state["messages"]
    message = messages[-1]
    if isinstance(message,AIMessage) and message.tool_calls:
        return "tools"
    else:
        return END

def main():
    workflow = StateGraph(MessagesState)
    workflow.add_node("node1",node1)
    workflow.add_node("tools",tools_node)
    workflow.add_edge(START,"node1")
    workflow.add_conditional_edges("node1",should_continue)
    workflow.add_edge("tools","node1")
    graph = workflow.compile()

    while True:
        msg = input("User: ")
        if msg.strip().lower() == "/q":
            print("good bye!")
            break;            
        chunks = graph.stream({"messages":[("user",msg)]})
        for chunk in chunks:
            for (key,value) in chunk.items():
                print("-"*10,key,"-"*10)
                message = value["messages"][-1]
                if isinstance(message,AIMessage) :
                    if message.tool_calls:
                        print("AI:")
                        print("    tool call: ",message.tool_calls)
                    else:
                        print("AI: ",message.content)
                else:
                    print("Tool: ",message.content)

if __name__ == "__main__":
    main()