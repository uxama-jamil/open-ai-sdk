# chainlit_app.py
import chainlit as cl
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, set_tracing_disabled, function_tool, ModelSettings

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

# ✅ Optional tool functions
@function_tool
def add(a: float, b: float) -> str:
    return f"{a} + {b} = {a + b}"

# Chainlit on_message hook
@cl.on_message
async def on_message(message: cl.Message):
    # Construct your agent
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant",
        model=model,
        # tools=[add],  # Enable if needed
        # model_settings=ModelSettings(tool_choice="required")
    )

    # Create a Chainlit message object to stream updates
    msg = cl.Message(content="")
    await msg.send()

    # Run the agent using Runner.run_streamed
    result = Runner.run_streamed(
        starting_agent=agent,
        input=message.content
    )

    async for event in result.stream_events():
        # Try to print delta/text/etc.
        if hasattr(event, "delta") and event.delta:
            msg.content += event.delta
            await msg.update()
        elif hasattr(event, "text") and event.text:
            msg.content += event.text
            await msg.update()
        elif hasattr(event, "message") and event.message and hasattr(event.message, "content"):
            msg.content += event.message.content
            await msg.update()

    # Optionally update final output if streaming didn’t work
    if not msg.content.strip():
        msg.content = result.final_output
        await msg.update()
