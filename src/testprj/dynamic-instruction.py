# Import core components from the openai-agents framework
import asyncio
from agents import Agent, ModelSettings, RunContextWrapper, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled
from agents.agent import StopAtTools
from dataclasses import dataclass

# Set up the external language model provider (Google's Gemini, using OpenAI-compatible API)
externalProvider = AsyncOpenAI(
    api_key="AIzaSyBc9lTEos9wzNpVcprceC9I1YfuQvNHmBk",  # API key for Gemini
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # Gemini-compatible endpoint
)


# Create a chat completions model using the Gemini model via the OpenAI API interface
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",  # Name of the Gemini model to use
    openai_client=externalProvider  # The provider configured above
)
set_tracing_disabled(True)

@dataclass
class DynamicInstructionAgent():
    style:str

def get_dynamic_instruction(wrapper:RunContextWrapper,agent: Agent):
    if wrapper.context.style == 'formal':
        return "You are a helpful assistant"
    elif wrapper.context.style == 'casual':
        return "You are a helpful assistant"
    elif wrapper.context.style == 'informal':
        return "Be rude and abusive"
    elif wrapper.context.style == 'sarcastic':
        return "Be sarcastic and funny"
    elif wrapper.context.style == 'poetic':
        return "Be poetic and flowery"
    elif wrapper.context.style == 'philosophical':
        return "Be philosophical and deep"
    elif wrapper.context.style == 'technical':
        return "Be technical and precise"
    elif wrapper.context.style == 'creative':
        return "Be creative and imaginative"

def runAgent():
    style = 'technical'
    instance = DynamicInstructionAgent(style=style)
    agent = Agent(
        name="Assistant",
        instructions=get_dynamic_instruction,
        model=model,
    )
    response = asyncio.run(Runner.run(
        starting_agent=agent,
        input="what is ur current role?",
        context = instance
    ))
    print(response.final_output)

runAgent()