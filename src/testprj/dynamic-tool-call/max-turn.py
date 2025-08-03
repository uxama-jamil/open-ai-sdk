# Import core components from the openai-agents framework
from dataclasses import dataclass
from datetime import datetime
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




@function_tool
def excersize_routine() -> str:
    """
    Returns the excersise routine and the suggestion  for which body part to excersise.
    """
    print(f"Excercising body part")
    return f"excersises for this week for body part."

@function_tool
def get_current_time() -> str:
    """
    Returns the current system time in ISO 8601 format.
    """
    return datetime.now().isoformat()

@function_tool
def product_suggestor(message: str) -> str:
    """
    Returns the product suggestion base on message.
    """
    print(f"Product suggestion for {message}")
    return f"product suggestion for {message} are......"

# Function to run a simple AI agent interaction
def runAgent():
    
    # Create an agent with a name and some instructions
    agent = Agent(
        name="Shopping assistent",
        instructions="You are a helpful assistant for shopping",
        model=model,  # Uses the Gemini model set above
        model_settings=ModelSettings( # tool_choice is required it means agent will use the tool onces afterward tool_choice will be set to auto
            tool_choice="required"
        ),
        # reset_tool_choice=False, #reset_tool_choice is by default true means it will changes tool_choice to auto after first tool call, 
                                   #on false it will not change tool_choice to auto
        tools=[get_current_time,excersize_routine,product_suggestor],
        # tool_use_behavior="stop_on_first_tool" # tool_use_behavior is stop_on_first_tool means agent will stop using tool after first tool call and return the tool result directly instead of passing result to llm
       
    )
   
    # Run the agent synchronously with a given input
    while True:
        input = input("Enter your message: ")
        if input == "exit" or input == "":
            break
        response = Runner.run_sync(
            starting_agent=agent,
            input=input,  # Input message to the assistant
            max_turns=2 #max_turns is the maximum number of turns the agent will take to complete the task
                        #llm receive a input either from user or from tool call is count as one turn
        )
        print(response.final_output)

   
    
runAgent()

