import asyncio
from agents import Agent, AgentHooks, ModelSettings, RunContextWrapper, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, Tool, function_tool, set_tracing_disabled,RunHooks
from agents.agent import StopAtTools
from dataclasses import dataclass

from dotenv import load_dotenv
import os
load_dotenv()

# Set up the external language model provider (Google's Gemini, using OpenAI-compatible API)
externalProvider = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),  # API key for Gemini
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # Gemini-compatible endpoint
)


# Create a chat completions model using the Gemini model via the OpenAI API interface
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",  # Name of the Gemini model to use
    openai_client=externalProvider  # The provider configured above
)
set_tracing_disabled(True)


@function_tool
def add(a: float, b: float) -> str:
    
    print("Tool add called")
    return f"{a} + {b} = {a + b}"

@function_tool
def subtract(a: float, b: float) -> str:
    
    print("Tool subtract called")
    return f"{a} - {b} = {a - b+1000}"

@function_tool
def multiply(a: float, b: float) -> str:
    
    print("Tool multiply called")
    return f"{a} * {b} = {a * b}"

@function_tool
def divide(a: float, b: float) -> str:
    
    print("Tool divide called")
    return "Error: Division by zero" if b == 0 else f"{a} / {b} = {a / b}"

class CustomRunnerHook(RunHooks):

    async def on_agent_start(self, context: RunContextWrapper, agent: Agent):
        print(f"Runner Agent started ")

    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: any):
        print(f"Runner Agent ended ")

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool):
        print(f"Runner Tool started  ")

    async def on_tool_end(self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str):
        print(f"Runner Tool ended ")

    async def on_handoff(self, context: RunContextWrapper, from_agent: Agent, to_agent: Agent):
        print(f"Runner Handoff")

class CustomAgentHook(AgentHooks):
    async def on_start(self, context: RunContextWrapper, agent: Agent):
        print(f"Agent started ")
    async def on_end(self, context: RunContextWrapper, agent: Agent, output: any):
        print(f"Agent ended ")
    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool):
        print(f"Agent Tool started  ")
    async def on_tool_end(self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str):
        print(f"Agent Tool ended ")
    async def on_handoff(self, context: RunContextWrapper, agent: Agent, source:Agent):
        print(f"Agent Handoff")

def runAgent():
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant",
        model=model,
        hooks=CustomAgentHook(),
        tools=[add,subtract,multiply,divide],
        model_settings=ModelSettings(
            tool_choice="required",
        )
    )
    response = asyncio.run(Runner.run(
        starting_agent=agent,
        input="what is ur current role and what is 1+1?",
        hooks=CustomRunnerHook()
    ))
    print(response.final_output)

runAgent()