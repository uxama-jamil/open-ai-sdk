import asyncio
from typing import Any
from agents import (Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, SQLiteSession)
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

"""
Example demonstrating session memory functionality.

This example shows how to use session memory to maintain conversation history
across multiple agent runs without manually handling .to_input_list().
"""



async def main():
    # Create an agent
    agent = Agent(
        name="Assistant",
        instructions="Reply very concisely.",
        model=model,
    )

    # Create a session instance that will persist across runs
    session_id = "conversation_123"
    session = SQLiteSession(session_id)

    print("=== Session Example ===")
    print("The agent will remember previous messages automatically.\n")

    # First turn
    while True:
        input_text = input("User: ")
        if input_text == "exit" or input_text == "":
            break
        result = await Runner.run(
            agent,
            input_text,
            session=session,
        )
        print(f"Assistant: {result.final_output}")
        print()

        # Second turn - the agent will remember the previous conversation
        
        

    print("=== Conversation Complete ===")
    print("Notice how the agent remembered the context from previous turns!")
    print("Sessions automatically handles conversation history.")

    # Demonstrate the limit parameter - get only the latest 2 items
    print("\n=== Latest Items Demo ===")
    latest_items = await session.get_items(limit=2)
    print("Latest 2 items:")
    for i, msg in enumerate(latest_items, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        print(f"  {i}. {role}: {content}")

    print(f"\nFetched {len(latest_items)} out of total conversation history.")

    # Get all items to show the difference
    all_items = await session.get_items()
    print(f"Total items in session: {len(all_items)}")


if __name__ == "__main__":
    asyncio.run(main())