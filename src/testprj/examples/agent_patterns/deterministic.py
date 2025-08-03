# Import core components from the openai-agents framework
import asyncio
from agents import Agent, AgentHooks, ModelSettings, RunContextWrapper, RunHooks, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, Tool, function_tool, set_tracing_disabled
from dotenv import load_dotenv
import os

from pydantic import BaseModel
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


"""
This example demonstrates a deterministic flow, where each step is performed by an agent.
1. The first agent generates a story outline
2. We feed the outline into the second agent
3. The second agent checks if the outline is good quality and if it is a scifi story
4. If the outline is not good quality or not a scifi story, we stop here
5. If the outline is good quality and a scifi story, we feed the outline into the third agent
6. The third agent writes the story
"""

class CustomAgentHook(AgentHooks):
    async def on_start(self, context: RunContextWrapper, agent: Agent):
        print(f"Agent started {agent.name} ")
    async def on_end(self, context: RunContextWrapper, agent: Agent, output: any):
        print(f"Agent ended {agent.name} and output is {output} ")
    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool):
        print(f"Agent Tool started {tool.name} ")
    async def on_tool_end(self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str):
        print(f"Agent Tool ended {tool.name} ")
    async def on_handoff(self, context: RunContextWrapper, agent: Agent, source:Agent):
        print(f"Agent Handoff {agent.name} to {source.name} ")
        
        
story_outline_agent = Agent(
    name="story_outline_agent",
    instructions="Generate a very short story outline based on the user's input.",
    model=model,
    hooks=CustomAgentHook(),
)


class OutlineCheckerOutput(BaseModel):
    good_quality: bool
    is_horror: bool



"""
    name : name of the agent
    Instructions : what the agent should do
    output_type: This tells the agent what kind of structured output you expect it to return
    model: (Gemini) 
"""
outline_checker_agent = Agent(
    name="outline_checker_agent",
    instructions="Read the given story outline, and judge the quality. Also, determine if it is a horror story.",
    output_type=OutlineCheckerOutput, #This tells the agent what kind of structured output you expect it to return
    model=model,
    hooks=CustomAgentHook(),
)

story_agent = Agent(
    name="story_agent",
    instructions="Write a short story based on the given outline.",
    output_type=str,#This tells the agent what kind of structured output you expect it to return
    model=model,
    hooks=CustomAgentHook(),
)
def main():
    input_prompt = input("What kind of story do you want? ")

    # Ensure the entire workflow is a single trace
    # with trace("Deterministic story flow"):
        # 1. Generate an outline
    outline_result = asyncio.run(Runner.run(
        story_outline_agent,
        input_prompt,
    ))
    print("///Outline generated///")

    # 2. Check the outline
    outline_checker_result =  asyncio.run(Runner.run(
        outline_checker_agent,
        outline_result.final_output,
    ))

    # 3. Add a gate to stop if the outline is not good quality or not a scifi story
    
    """
    assert isinstance(outline_checker_result.final_output, OutlineCheckerOutput)
    This is a safety check to make sure the output is actually of the type you expected.
        If the model returned something invalid or unstructured, this would raise an error instead of continuing with bad data.
    """
    assert isinstance(outline_checker_result.final_output, OutlineCheckerOutput)
    if not outline_checker_result.final_output.good_quality:
        print("Outline is not good quality, so we stop here.")
        exit(0)

    if not outline_checker_result.final_output.is_horror:
        print("Outline is not a horror story, so we stop here.")
        exit(0)

    print("///Outline is good quality and a horror story, so we continue to write the story.///")

    # 4. Write the story
    story_result =  asyncio.run(Runner.run(
        story_agent,
        outline_result.final_output,
    ))
    print(f"Story: {story_result.final_output}")


if __name__ == "__main__":
    main()