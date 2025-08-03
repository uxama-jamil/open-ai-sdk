import asyncio
from typing import Any, Literal
from agents import (Agent, AgentHooks, FunctionToolResult, ModelSettings, RunContextWrapper, RunHooks, 
                    Runner, AsyncOpenAI, OpenAIChatCompletionsModel, Tool, ToolsToFinalOutputFunction, 
                    ToolsToFinalOutputResult, function_tool, set_tracing_disabled)
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
This example shows how to force the agent to use a tool. It uses `ModelSettings(tool_choice="required")`
to force the agent to use any tool.

You can run it with 3 options:
1. `default`: The default behavior, which is to send the tool output to the LLM. In this case,
    `tool_choice` is not set, because otherwise it would result in an infinite loop - the LLM would
    call the tool, the tool would run and send the results to the LLM, and that would repeat
    (because the model is forced to use a tool every time.)
2. `first_tool_result`: The first tool result is used as the final output.
3. `custom`: A custom tool use behavior function is used. The custom function receives all the tool
    results, and chooses to use the first tool result to generate the final output.

Usage:
python examples/agent_patterns/forcing_tool_use.py -t default
python examples/agent_patterns/forcing_tool_use.py -t first_tool
python examples/agent_patterns/forcing_tool_use.py -t custom

or 

python examples/agent_patterns/forcing_tool_use.py --usama-jamil default
python examples/agent_patterns/forcing_tool_use.py --usama-jamil first_tool
python examples/agent_patterns/forcing_tool_use.py --usama-jamil custom
"""


class Weather(BaseModel):
    city: str
    temperature_range: str
    conditions: str


@function_tool
def get_weather(city: str) -> Weather:
    print("[debug] get_weather called")
    return Weather(city=city, temperature_range="14-20C", conditions="Sunny with wind")


async def custom_tool_use_behavior(
    context: RunContextWrapper[Any], results: list[FunctionToolResult]
) -> ToolsToFinalOutputResult:
    weather: Weather = results[0].output
    return ToolsToFinalOutputResult(
        is_final_output=True, final_output=f"{weather.city} is {weather.conditions}."
    )

""" 
    tool_use_behavior = "run_llm_again" = Let the LLM handle the tool result
        -The LLM will read the tool result and write a full sentence like:
            -"The weather in Tokyo is Sunny with wind and the temperature ranges from 14 to 20°C."
        🧠 Use this when you want the model to reason or reword the tool’s response.
        
    tool_use_behavior = "stop_on_first_tool" = Just return the tool result directly
                Whatever the tool returns (like a Weather object) is used as-is.
                You skip the LLM and show the raw result to the user.
        🔴 Risk: May not be human-readable (unless the tool returns a string).
                  
    tool_use_behavior = ToolsToFinalOutputResult() =  Use your own Python function to finalize the output
            -You’re formatting the tool result yourself
            -You can use this when you want to use your own Python function to finalize the output.
        ✅ Best for full control and custom formatting
    
"""

def main(tool_use_behavior: Literal["default", "first_tool", "custom"] = "default"):
    if tool_use_behavior == "default":
        behavior: Literal["run_llm_again", "stop_on_first_tool"] | ToolsToFinalOutputFunction = (
            "run_llm_again"
        )
    elif tool_use_behavior == "first_tool":
        behavior = "stop_on_first_tool"
    elif tool_use_behavior == "custom":
        behavior = custom_tool_use_behavior

    agent = Agent(
        name="Weather agent",
        instructions="You are a helpful agent.",
        tools=[get_weather],
        tool_use_behavior=behavior,
        model_settings=ModelSettings(
            tool_choice="required" if tool_use_behavior != "default" else None
        ),
        model=model,
    )

    result = asyncio.run(Runner.run(agent, input="What's the weather in karachi?"))
    print(result.final_output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--usama-jamil",
        type=str,
        required=True,
        choices=["default", "first_tool", "custom"],
        help="The behavior to use for tool use. Default will cause tool outputs to be sent to the model. "
        "first_tool_result will cause the first tool result to be used as the final output. "
        "custom will use a custom tool use behavior function.",
    )
    args = parser.parse_args()
    main(args.usama_jamil)