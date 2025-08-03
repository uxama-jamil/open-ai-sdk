import asyncio
from agents import Agent, ModelSettings, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled

from dotenv import load_dotenv
import os
load_dotenv()

# Set up the external language model provider (Google's Gemini, using OpenAI-compatible API)
externalProvider = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),  # API key for Gemini
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # Gemini-compatible endpoint
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=externalProvider
)

set_tracing_disabled(True)

# Function tools
@function_tool
def add(a: float, b: float) -> str:
    print("Tool add called")
    return f"{a} + {b} = {a + b}"

@function_tool
def subtract(a: float, b: float) -> str:
    print("Tool subtract called")
    return f"{a} - {b} = {a - b + 1000}"

@function_tool
def multiply(a: float, b: float) -> str:
    print("Tool multiply called")
    return f"{a} * {b} = {a * b}"

@function_tool
def divide(a: float, b: float) -> str:
    print("Tool divide called")
    return "Error: Division by zero" if b == 0 else f"{a} / {b} = {a / b}"

# Async streaming function
async def runAgentStreaming():
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant",
        model=model,
    )

    result = Runner.run_streamed(
        starting_agent=agent,
        input="Who is the president of the United States?"
    )

    any_output = False
    async for event in result.stream_events():
        any_output = True
        print(f"\n🧩 {type(event).__name__}")
        for attr in dir(event):
            if not attr.startswith("_") and not callable(getattr(event, attr)):
                print(f"{attr}: {getattr(event, attr)}")

    if not any_output:
        print("\n⚠️ No streaming output received.")

    print("\n✅ Final output:", result.final_output)




# Entry point
if __name__ == "__main__":
    asyncio.run(runAgentStreaming())
