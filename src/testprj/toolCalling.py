# Import core components from the openai-agents framework
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled,function_tool
import requests
import asyncio

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
def get_current_weather(city: str) -> str:
    """
    Fetches the current weather for a given city using the OpenWeatherMap API.

    Args:
        city (str): Name of the city (or country capital) to get weather for.
        api_key (str): Your OpenWeatherMap API key.

    Returns:
        str: Weather description and temperature in Celsius, or an error message.
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': '1d7527df97376b8cd1cec4882abe27f8',
        'units': 'metric'  # Use 'imperial' for Fahrenheit
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code == 200:
            weather = data['weather'][0]['description'].capitalize()
            temp = data['main']['temp']
            return f"The current weather in {city} is {weather} with a temperature of {temp}°C."
        else:
            return f"Error: {data.get('message', 'Unable to fetch weather.')}"
    except Exception as e:
        return f"Exception occurred: {str(e)}"


@function_tool
def get_addition(a: int, b: int) -> int:
    """
    Returns the sum of two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The addition of the two input integers.

    Example:
        >>> get_addition(3, 5)
        8
    """
    print(f"testing tools addition")
    return f"The addition of {a} and {b} is {a+b}"

# Tool 1: Addition
@function_tool
def add(a: float, b: float) -> str:
    """Returns the sum of two numbers."""
    print(f"testing tools addition")
    return f"{a} + {b} = {a + b}"

# Tool 2: Subtraction
@function_tool
def subtract(a: float, b: float) -> str:
    """Returns the result of subtracting b from a."""
    print(f"testing tools subtraction")
    return f"{a} - {b} = {a - b}"

# Tool 3: Multiplication
@function_tool
def multiply(a: float, b: float) -> str:
    """Returns the product of two numbers."""
    print(f"testing tools multiplication")
    return f"{a} * {b} = {a * b}"

# Tool 4: Division
@function_tool
def divide(a: float, b: float) -> str:
    """Returns the result of dividing a by b."""
    print(f"testing tools division")
    if b == 0:
        return "Error: Division by zero"
    return f"{a} / {b} = {a / b}"


# Disable internal tracing/logging (optional – for performance or privacy)
set_tracing_disabled(True)

# Function to run a simple AI agent interaction
def runAgent():
    # Create an agent with a name and some instructions
    print(f"running agent")
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant",
        model=model,  # Uses the Gemini model set above
        tools=[add,subtract] # This is the tool that will be used to get the temperature of the city
    )

    # Run the agent synchronously with a given input
    response = Runner.run_sync(
        starting_agent=agent,
        input="What is the result of the 1+1-2"  # Input message to the assistant
    )

    # Print the agent's final response
    print(response.final_output)
    
    
    
async def  runAgentAsync ():
    # Create an agent with a name and some instructions
    print(f"running agent")
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant",
        model=model,  # Uses the Gemini model set above
        tools=[add,subtract,multiply,divide,get_current_weather] # This is the tool that will be used to get the temperature of the city
    )

    # Run the agent synchronously with a given input
    response = await Runner.run(
        starting_agent=agent,
        input="What is the result of the 1+1-200*200/50? and what is the current weather in karachi and lahore?"  # Input message to the assistant
    )

    # Print the agent's final response
    print(response.final_output)
    
def testAsync():
    asyncio.run(runAgentAsync())


