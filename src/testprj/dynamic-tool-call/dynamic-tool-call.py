# Import core components from the openai-agents framework
from dataclasses import dataclass
from agents import Agent, ModelSettings, RunContextWrapper, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled
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

# Disable internal tracing/logging (optional – for performance or privacy)
set_tracing_disabled(True)

@dataclass
class user:
    is_user_logged_in: bool

def is_tool_callable(context:RunContextWrapper , agent:Agent): # ithis function that checks if the tool is callable if it's true the tool will be enabled
    print(f"Checking if tool is callable: {context}")
    if context.context.is_user_logged_in:
        return True
    else:
        return False

@function_tool(is_enabled=is_tool_callable)  # is_enabled is a function that checks if the tool is callable if it's true the tool will be enabled
def get_current_weather(city: str) -> str:
    """Get the current weather for a given city."""
    print(f"Getting weather for {city}")
    return f"The weather in {city} is sunny."

# Function to run a simple AI agent interaction
def runAgent():
    """
    Creates and runs an AI assistant agent using Gemini (OpenAI-compatible).
    Prints the final output from the conversation.
    
    Example:
        >>> runAgent()
        The capital of a black hole is not defined...
    """
    # Create an agent with a name and some instructions
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant and you can use the get_current_weather tool to get the current weather for a given city.",
        model=model,  # Uses the Gemini model set above
        tools=[get_current_weather],
        model_settings=ModelSettings(
            tool_choice="required"
        )
    )
    obj = user(is_user_logged_in=True)
    # Run the agent synchronously with a given input
    response = Runner.run_sync(
        starting_agent=agent,
        input="What is the weather in Tokyo?",  # Input message to the assistant
        context=obj
    )

    # Print the agent's final response
    print(response.final_output)
    
runAgent()

