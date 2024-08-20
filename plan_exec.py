import asyncio
from datetime import datetime
import operator
from youht_app1.langchain import LangchainLib
from langgraph.graph import START,END,StateGraph
from langgraph.graph.message import MessagesState
from langchain.prompts import StringPromptTemplate,PromptTemplate,ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode
from typing import Annotated, List, Literal, TypedDict
from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel,Field
from langchain_core.messages import ToolMessage,HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

langchainLib = LangchainLib()
llm = langchainLib.llmLib.get_llm("DEEPBRICKS_API_KEY","gpt-4o-mini","https://api.deepbricks.ai/v1")
#llm = langchainLib.llmLib.get_llm("GROQ_API_KEY","llama-3.1-70b-versatile","https://api.groq.com/openai/v1")
#llm = langchainLib.llmLib.get_llm("GROQ_API_KEY","llama3-groq-70b-8192-tool-use-preview","https://api.groq.com/openai/v1")
checkpointer = MemorySaver()
    

repl = PythonREPLTool()

@tool
def web_search(query:str):
    '''
    从互联网搜索query问题
    '''
    duckduckgo = DuckDuckGoSearchRun()
    res =  duckduckgo.invoke(query)
    print(res)
    return res

tools = [web_search]

tools_node = ToolNode(tools=tools) 

llm_with_tools = llm.bind_tools(tools,strict=True)

class Plan(BaseModel):
    """未来要执行的计划"""
    steps: List[str] = Field(description="different steps to follow,should be in sorted order")

class PlanExecState(MessagesState):
    input: str
    plan: List[str]
    past_steps: Annotated[List[tuple],operator.add] 
    response: str

def should_continue(state:PlanExecState)->Literal["tools","execute_node","__end__"]:
    plan = state["plan"]
    past_steps = state["past_steps"]
    if past_steps:
        past_step = past_steps[-1]
        if past_step[1].tool_calls:
            print("goto tools.........................")
            return "tools"
        elif len(plan)==0:
            return END
    return "execute_node"
     
def plan_node(state:PlanExecState):
    plan_prompt = ChatPromptTemplate.from_template(
"""
现在的时间是:
{today}
你是AI助手，你需要为给定的目标指定一个简单的逐步计划。
这个计划应该包括单个任务，如果正确执行，将会得到正确的答案。不要添加任何多余的步骤。
最后一步的结果应该是最终的答案。确保每个步骤都有所需的所有信息 - 不要跳过步骤。
{objective}
"""
    ).partial(today=datetime.now())
    input = state["input"]
    chain = plan_prompt | llm.with_structured_output(Plan)
    res:Plan = chain.invoke({"objective":input})
    res.steps.reverse()
    return {"plan":res.steps}

def execute_node(state:PlanExecState):
    execute_prompt = ChatPromptTemplate.from_template(
"""
根据上下文的提示，得出进一步答案,必要的时候可以使用网络搜索的工具。
context:
{context}
question:
{question}
"""
    )
    messages = state["messages"]
    context = ""
    if messages:
        message = messages[-1]
        print("*"*50,len(messages),type(message))
        if isinstance(message,ToolMessage):
            #print("tool message",message.content)
            context = message.content
    question = state["input"]
    plan = state["plan"]
    past_steps = state["past_steps"]
    #print("plan,past_steps------>",plan,past_steps)
    #context = "\n".join([item[1].content for item in past_steps])
    if len(plan)>0:
        question = plan.pop()
    chain = execute_prompt | llm_with_tools
    res = chain.invoke({"context":context,"question":question})
    #print("----->",question,res)    
    return {"plan":plan,"past_steps":[(question,res)],"messages":[res]}    

def replan_node(state:PlanExecState):
    replan_prompt = ChatPromptTemplate.from_template(
"""
你是AI助手，你可以使用工具帮助用户。
你需要检查目前已经执行的步骤>>{past_steps}<< 是否回答了用户的问题。
如果能够回答用户的问题，则返回结果;如果不能回答用户的问题，进行下面的步骤：
对于给定的目标，指定一个简单的逐步计划。
这个计划应该包括单个任务，如果正确执行，将会得到正确的答案。不要添加任何多余的步骤。
最后一步的结果应该是最终的答案。确保每个步骤都有所需的所有信息 - 不要跳过步骤。
现在的时间是:
{today}
你的目标是：
{input}
你最初的计划是：
{plan}
你目前已经完成了以下步骤：
{past_steps}
请相应地更新你的计划。如果不需要更多步骤并且可以返回给用户，那么就这样回应。否则，请填写计划。只添加仍需要完成的步骤到计划中。
不要将已经完成的步骤添加到计划中。
不要将已完成的步骤作为计划的一部分返回
"""        
    ).partial(today=datetime.now())

def should_end(state:PlanExecState):
    if "response" in state and state["response"]:
        return True
    else:
        return False
    
async def main():
    workflow = StateGraph(PlanExecState)
    workflow.add_node("plan_node",plan_node)
    workflow.add_node("execute_node",execute_node)
    workflow.add_node("tools",ToolNode(tools))
    workflow.add_edge(START,"plan_node")
    workflow.add_edge("plan_node","execute_node")
    workflow.add_conditional_edges("execute_node",should_continue)
    workflow.add_edge("tools","execute_node")
    workflow.add_edge("execute_node",END)

    graph = workflow.compile(checkpointer=checkpointer)
    
    chunks = graph.astream({"input":"今年美国总统候选人年龄的总和是多少？"},config={"configurable": {"thread_id": 42}})
    async for chunk in chunks:
        for key,value in chunk.items():
            print("----",key,"-----")
            print(value)

if __name__ == "__main__":
    asyncio.run(main())