import asyncio
from typing import Any, Literal
import uuid
from agents import (Agent, AgentHooks, FunctionToolResult, ModelSettings, RawResponsesStreamEvent, RunContextWrapper, RunHooks, 
                    Runner, AsyncOpenAI, OpenAIChatCompletionsModel, TResponseInputItem, Tool, ToolsToFinalOutputFunction, 
                    ToolsToFinalOutputResult, function_tool, set_tracing_disabled)
from dotenv import load_dotenv
import os

from pydantic import BaseModel
load_dotenv()

# Set up the external language model provider (Google's Gemini, using OpenAI-compatible API)
externalProvider = AsyncOpenAI(
    api_key="ollama",  # dummy value to satisfy SDK
    base_url="http://localhost:11434/v1"  # local Ollama endpoint
)

# Create a chat completions model using the Gemini model via the OpenAI API interface
model = OpenAIChatCompletionsModel(
    model="llama3.2:latest",  # Name of the llama model to use from ollama
    openai_client=externalProvider  # The provider configured above
)

# Disable internal tracing/logging (optional – for performance or privacy)
set_tracing_disabled(True)

"""
This example shows the handoffs/routing pattern. The triage agent receives the first message, and
then hands off to the appropriate agent based on the language of the request. Responses are
streamed to the user.
"""

@function_tool(name_override="return_to_triage")
def return_to_triage() -> str:
    """Use this tool when the user wants to go back to the main triage agent or switch to a different language/agent."""
    print("return_to_triage tool called")
    return "Returning to triage agent"

@function_tool(name_override="multiply")
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    print(f"multiply tool called with {a} and {b}")
    return a * b

def should_switch_to_triage(message: str) -> bool:
    """Check if user wants to switch languages"""
    switch_keywords = [
        "switch language", "change language", "cambiar idioma", 
        "changer de langue", "main menu", "menu principal",
        "go back", "volver", "retourner", "different language"
    ]
    return any(keyword in message.lower() for keyword in switch_keywords)

french_agent = Agent(
    name="french_agent",
    instructions="""You only speak French. If the user asks to go back to the main menu, 
    switch languages, or talk to a different agent, use the return_to_triage tool.""",
    model=model,
    tools=[return_to_triage],
)

spanish_agent = Agent(
    name="spanish_agent",
    instructions="""You only speak Spanish. If the user asks to go back to the main menu, 
    switch languages, or talk to a different agent, use the return_to_triage tool.""",
    model=model,
    tools=[return_to_triage],
)

english_agent = Agent(
    name="english_agent",
    instructions="""You only speak English. If the user asks to go back to the main menu, 
    switch languages, or talk to a different agent, use the return_to_triage tool.""",
    model=model,
    tools=[return_to_triage, multiply],
)

triage_agent = Agent(
    name="triage_agent",
    instructions="""You are the main triage agent. Handoff to the appropriate agent based on 
    the language or request.
    Welcome users and help them choose the right language agent.""",
    handoffs=[french_agent, spanish_agent, english_agent],
    model=model,
    
)


async def main():
    msg = input("Hi! We speak French, Spanish and English. How can I help? ")
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        # Each conversation turn is a single trace. Normally, each input from the user would be an
        # API request to your app, and you can wrap the request in a trace()
        print(f"Current agent is {agent.name}")
        result = Runner.run_streamed(
            agent,
            input=inputs,
        )
        # Track if we need to switch back to triage
        should_switch_to_triage_flag = False
        
        async for event in result.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                # Handle text streaming from the model
                if hasattr(event.data, 'delta') and event.data.delta:
                    print(event.data.delta, end="", flush=True)
                elif hasattr(event.data, 'content') and event.data.content:
                    print(event.data.content, end="", flush=True)
            elif hasattr(event, 'data') and isinstance(event.data, FunctionToolResult):
                # Check if the switch_language tool was called
                if event.data.function_name == "switch_language" and event.data.result == "SWITCH_TO_TRIAGE":
                    should_switch_to_triage_flag = True
                    print("\nSwitching back to main menu...\n")

        inputs = result.to_input_list()
        print("\n")

        # If switch was requested via tool and we're not already on triage, go back to triage agent
        if should_switch_to_triage_flag and agent.name != "triage_agent":
            agent = triage_agent
            inputs.append({"content": "I'd like to switch languages or go back to the main menu.", "role": "user"})
        else:
            user_msg = input("Enter a message (or type 'switch language' to change): ")
            
            # Manual check for switch request - only if not already on triage
            if should_switch_to_triage(user_msg) and agent.name != "triage_agent":
                agent = triage_agent
                inputs.append({"content": "I'd like to switch languages.", "role": "user"})
            else:
                inputs.append({"content": user_msg, "role": "user"})
                agent = result.current_agent
        print(f"Current agent is {agent.name}...")


if __name__ == "__main__":
    asyncio.run(main())