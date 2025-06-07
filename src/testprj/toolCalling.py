# Import core components from the openai-agents framework
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled,function_tool

# Set up the external language model provider (Google's Gemini, using OpenAI-compatible API)
externalProvider = AsyncOpenAI(
    api_key="AIzaSyB4BU0F5fGuXkkoWx5xhal-X9w81BZYcHw",  # API key for Gemini
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # Gemini-compatible endpoint
)

# Create a chat completions model using the Gemini model via the OpenAI API interface
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",  # Name of the Gemini model to use
    openai_client=externalProvider  # The provider configured above
)

#This is a tool that returns the temperature of the city
@function_tool
def get_temperature(city:str):
    """
    Retrieve the current temperature for a given city.

    Parameters:
        city (str): The name of the city for which to retrieve the temperature.

    Returns:
        str: A string describing the temperature in the specified city.
    
    Example:
        >>> get_temperature("Paris")
        'The temperature in Paris is 20 degrees Celsius'
    """
    return f"The temperature in {city} is 20 degrees Celsius"

# Disable internal tracing/logging (optional – for performance or privacy)
set_tracing_disabled(True)

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
        instructions="You are a helpful assistant",
        model=model,  # Uses the Gemini model set above
        tools=[get_temperature] # This is the tool that will be used to get the temperature of the city
    )

    # Run the agent synchronously with a given input
    response = Runner.run_sync(
        starting_agent=agent,
        input="What is the capital of blackHole?"  # Input message to the assistant
    )

    # Print the agent's final response
    print(response.final_output)

