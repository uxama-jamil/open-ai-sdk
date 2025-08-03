from dataclasses import dataclass
from agents import Agent, AgentHooks, RunContextWrapper, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, TContext, Tool, set_tracing_disabled,function_tool,handoff
from pydantic import BaseModel
import requests
import asyncio

from agents import enable_verbose_stdout_logging # for debugging

enable_verbose_stdout_logging() # for debugging

set_tracing_disabled(True)
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
    model="gemini-2.0-flash",  # Name of the Gemini model to use
    openai_client=externalProvider  # The provider configured above
)


class PhysicsAnswer(BaseModel):
    is_physic_homework: bool
    reasoning: str

physics_agent = Agent(
    name="Physics Agent",
    instructions="You are a physics agent. You are given a question and you need to answer it based on the physics.",
    output_type=PhysicsAnswer,
    model=model,
)

physics_tool = physics_agent.as_tool(tool_name="physics_tool",tool_description="This tool is used to get the physics context of the question")

agent = Agent(
    name="Front Agent",
    instructions="You are a main agent. You are given a question and you need to answer it based on the physics.",
    model=model,
    tools=[physics_tool],
)


# class MathHomeworkOutput(BaseModel):
#     is_math_homework: bool
#     reasoning: str

# guardrail_agent = Agent( 
#     name="Guardrail check",
#     instructions="Check if the user is asking you to do their math homework.",
#     model=model,
#     output_type=MathHomeworkOutput,
# )


class UrduAnswer(BaseModel):
    is_urdu_question: bool
    urdu_answer: str
    reasoning: str
urdu_agent = physics_agent.clone(
    name="Urdu Agent",
    instructions="You are a urdu agent. You are given a question and you need to answer it based on the urdu.",
    output_type=UrduAnswer,
)

urdu_tool = urdu_agent.as_tool(tool_name="urdu_tool",tool_description="This tool is used to get the urdu context of the question")


front_agent = Agent(
    name="Front Agent",
    instructions="""
    **TASK**: Handle Urdu translation requests
    **PROCEDURE**:
    1. If user asks for Urdu translation:
       - Use urdu_translator tool ONCE
       - Return tool's EXACT output without modification
    2. NEVER add:
       - Extra text like "The translation is:"
       - Your own comments
       - English explanations
    3. Output MUST be RAW tool response
    
    **EXAMPLE**:
    Input: "Good morning, translate to Urdu"
    Output: {"is_urdu_answer": true, "reasoning": "صبح بخیر"}
    """,
    model=model,
    tools=[urdu_tool],
)

response = asyncio.run(Runner.run(
    starting_agent=front_agent,
    input="what is ur name , translate this in urdu?",

))
# Print execution trace
print(response.final_output)
# if isinstance(response.final_output, PhysicsAnswer):
#     print(f"Is Physics Question: {response.final_output.is_physics_question}")
#     print(f"Physics Answer: {response.final_output.answer}")
# else:
#     print(f"Front Agent's direct response: {response.final_output} and response history {response.history}")
# print(response.final_output)
# print(response.last_agent.name)

 # Access the key of the dictionary


