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
class User:
    user_type: str

def is_tool_callable(ctxt:RunContextWrapper , agent:Agent):
    print(f"Checking if tool is callable: {ctxt}")
    if ctxt.context.user_type == "premium":
        return True
    return False



@function_tool
def multiply(number: int) -> str:
    """Multiply the number by 2."""
    print(f"Multiplying {number} by 2")
    return f"processed number  {number*2}." 

@function_tool
def add(number: int) -> str:
    """Add 10 to the number."""
    print(f"Adding 10 to {number}")
    return f"processed number  {number+10}." 


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
        name="Calculator",
        instructions="You are a helpful assistant and you are a calculator.",
        model=model,  # Uses the Gemini model set above
        model_settings=ModelSettings(
            tool_choice="required"
        ),
        # reset_tool_choice=False,
        tools=[add,multiply],
       
    )
    obj = User(user_type="free")
    # Run the agent synchronously with a given input
    response = Runner.run_sync(
        starting_agent=agent,
        input="process number 5?",  # Input message to the assistant
    )

    # Print the agent's final response
    print(response.final_output)
    
runAgent()

