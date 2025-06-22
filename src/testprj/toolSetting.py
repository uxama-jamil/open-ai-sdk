# Import core components from the openai-agents framework
import asyncio
from agents import Agent, ModelSettings, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled
from agents.agent import StopAtTools

# Set up the external language model provider (Google's Gemini, using OpenAI-compatible API)
externalProvider = AsyncOpenAI(
    api_key="AIzaSyBc9lTEos9wzNpVcprceC9I1YfuQvNHmBk",  # API key for Gemini
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # Gemini-compatible endpoint
)
# externalProvider = AsyncOpenAI(  # API key for Gemini
#     api_key="1234567890",
#     base_url="http://localhost:11434/api/chat",  # Gemini-compatible endpoint
# )

# Create a chat completions model using the Gemini model via the OpenAI API interface
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",  # Name of the Gemini model to use
    openai_client=externalProvider  # The provider configured above
)

# Disable internal tracing/logging (optional – for performance or privacy)
set_tracing_disabled(True)

# Global counter
tool_call_count = {
    'add': 0,
    'subtract': 0,
    'multiply': 0,
    'divide': 0
}

@function_tool
def add(a: float, b: float) -> str:
    tool_call_count['add'] += 1
    print("Tool add called")
    return f"{a} + {b} = {a + b}"

@function_tool
def subtract(a: float, b: float) -> str:
    tool_call_count['subtract'] += 1
    print("Tool subtract called")
    return f"{a} - {b} = {a - b+1000}"

@function_tool
def multiply(a: float, b: float) -> str:
    tool_call_count['multiply'] += 1
    print("Tool multiply called")
    return f"{a} * {b} = {a * b}"

@function_tool
def divide(a: float, b: float) -> str:
    tool_call_count['divide'] += 1
    print("Tool divide called")
    return "Error: Division by zero" if b == 0 else f"{a} / {b} = {a / b}"


# @function_tool
# def get_weather(city: str) -> str:
#     """Returns the weather of a city."""
#     print(f"testing tools weather")
#     return f"The weather of {city} is sunny"


# Function to run a simple AI agent interaction
def runAgent():
    
    # Create an agent with a name and some instructions
    print(f"running sync agent")
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant",
        model=model,  # Uses the Gemini model set above
        tools=[add, subtract, multiply, divide],
        model_settings=ModelSettings(
            tool_choice="required",
        )
    )

    # Run the agent synchronously with a given input
    response = Runner.run_sync(
        starting_agent=agent,
        input="What is the capital of blackHole?"  # Input message to the assistant
    )

    # Print the agent's final response
    print(response.final_output)
    
def  runAgentAsync ():
    # Create an agent with a name and some instructions
    print(f"running asyncagent")
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant",
        model=model,  # Uses the Gemini model set above
        tools=[subtract,add,multiply,divide], # This is the tool that will be used to get the temperature of the city
        model_settings=ModelSettings(
            tool_choice="required",
        )
        
    )

    # Run the agent synchronously with a given input
    response = asyncio.run(Runner.run(
        max_turns=2,
        starting_agent=agent,
        input="what is 1+1?"  # Input message to the assistant
    ))
    print('tool_call_count',tool_call_count)

    # Print the agent's final response
    print(response.final_output)
    
def  runAgentAsyncWithoutllmVerification ():
    # Create an agent with a name and some instructions
    print(f"running asyncagent")
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant",
        model=model,  # Uses the Gemini model set above
        tools=[subtract,add,multiply,divide], # This is the tool that will be used to get the temperature of the city
        tool_use_behavior='stop_on_first_tool', #llm will not verify the tool call result and will stop at the first tool call
        
    )

    # Run the agent synchronously with a given input
    response = asyncio.run(Runner.run(
        max_turns=2,
        starting_agent=agent,
        input="what is 2-1?"  # Input message to the assistant
    ))
    print('tool_call_count',tool_call_count)
    # running asyncagent
    # Tool subtract called
    # tool_call_count {'add': 0, 'subtract': 1, 'multiply': 0, 'divide': 0}
    # 2.0 - 1.0 = 1001.0  #this is the result of the first tool call

    # Print the agent's final response
    print(response.final_output)
    
runAgentAsyncWithoutllmVerification()
    