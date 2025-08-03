# Import core components from the openai-agents framework
from dataclasses import dataclass
from agents import Agent, ModelSettings, RunContextWrapper, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled

# Set up the external language model provider (Google's Gemini, using OpenAI-compatible API)
externalProvider = AsyncOpenAI(
    api_key="AIzaSyB8CcLUoNkWeNvcOGKxFgGjhUEW4eoCveE",  # API key for Gemini
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



@function_tool(is_enabled=is_tool_callable) 
def get_discount(price: int) -> str:
    """Get the discount for a given price."""
    print(f"Getting discount for {price}")
    return f"Your final price after discount is {price*0.9}."

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
        name="Sales Manager",
        instructions="You are a helpful assistant and you are a sales manager.",
        model=model,  # Uses the Gemini model set above
        tools=[get_discount],
       
    )
    obj = User(user_type="free")
    # Run the agent synchronously with a given input
    response = Runner.run_sync(
        starting_agent=agent,
        input="here is my bill 500$ is there any discount available?",  # Input message to the assistant
        context=obj
    )

    # Print the agent's final response
    print(response.final_output)
    
runAgent()

