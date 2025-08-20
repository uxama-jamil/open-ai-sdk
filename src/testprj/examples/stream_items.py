import asyncio
import random
from typing import Any
from agents import (Agent, ItemHelpers, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled, SQLiteSession)
from dotenv import load_dotenv

load_dotenv()

# Set up the external language model provider (Ollama)
externalProvider = AsyncOpenAI(
    api_key="ollama",  # dummy value to satisfy SDK
    base_url="http://localhost:11434/v1"  # local Ollama endpoint
)

# Create a chat completions model using the Llama vision model via the OpenAI API interface
model = OpenAIChatCompletionsModel(
    model="llama3.2:latest",  # Use the vision-specific model
    openai_client=externalProvider
)

set_tracing_disabled(True)

@function_tool
def how_many_jokes() -> int:
    """Return a random number of jokes between 1 and 10"""
    print("how_many_jokes tool called")
    return random.randint(1, 10)


async def main():
    agent = Agent(
        name="Joker",
        instructions="""First call the `how_many_jokes` tool, then tell that number of hilarious jokes.
        Note: You must call the tool first, then use the output to tell that number of jokes and share those all jokes instead of asking .
        """,
        tools=[how_many_jokes],
        model=model,
    )

    result = Runner.run_streamed(
        agent,
        input="Hello",
    )
    print("=== Run starting ===")
    async for event in result.stream_events():
        # We'll ignore the raw responses event deltas
        if event.type == "raw_response_event":
            continue
        elif event.type == "agent_updated_stream_event":
            print(f"Agent updated: {event.new_agent.name}")
            continue
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                print("-- Tool was called")
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}")
            elif event.item.type == "message_output_item":
                print(f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}")
            else:
                pass  # Ignore other event types

    print("=== Run complete ===")


if __name__ == "__main__":
    asyncio.run(main())