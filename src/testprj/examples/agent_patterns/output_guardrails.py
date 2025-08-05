import asyncio
import json
from typing import Any, Literal
from agents import (Agent,  GuardrailFunctionOutput, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered, RunContextWrapper, 
                    Runner, AsyncOpenAI, OpenAIChatCompletionsModel, TResponseInputItem, Tool, ToolsToFinalOutputFunction, 
                     function_tool, input_guardrail, output_guardrail, set_tracing_disabled)
from dotenv import load_dotenv
import os


from pydantic import BaseModel, Field
load_dotenv()

# Set up the external language model provider (Google's Gemini, using OpenAI-compatible API)
externalProvider = AsyncOpenAI(
    api_key="ollama",  # dummy value to satisfy SDK
    base_url="http://localhost:11434/v1"  # local Ollama endpoint
)

# Create a chat completions model using the Gemini model via the OpenAI API interface
model = OpenAIChatCompletionsModel(
    model="llama3.2:latest",  # Name of the llama model to use from ollama
    openai_client=externalProvider  # The provider configured above
)

# Disable internal tracing/logging (optional – for performance or privacy)
set_tracing_disabled(True)


"""
This example shows how to use output guardrails.

Output guardrails are checks that run on the final output of an agent.
They can be used to do things like:
- Check if the output contains sensitive data
- Check if the output is a valid response to the user's message

In this example, we'll use a (contrived) example where we check if the agent's response contains
a phone number.
"""


# The agent's output type
class MessageOutput(BaseModel):
    reasoning: str = Field(description="Thoughts on how to respond to the user's message")
    response: str = Field(description="The response to the user's message")
    user_name: str | None = Field(description="The name of the user who sent the message, if known")


@output_guardrail
async def sensitive_data_check(
    context: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    phone_number_in_response = "650" in output.response
    phone_number_in_reasoning = "650" in output.reasoning

    return GuardrailFunctionOutput(
        output_info={ #guardrail output if it trips
            "phone_number_in_response": phone_number_in_response,
            "phone_number_in_reasoning": phone_number_in_reasoning,
        },
        tripwire_triggered=phone_number_in_response or phone_number_in_reasoning,
    )


agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    output_type=MessageOutput,
    output_guardrails=[sensitive_data_check],
    model=model,
)


async def main():
    # This should be ok
    await Runner.run(agent, "What's the capital of California?")
    print("First message passed")

    # This should trip the guardrail
    try:
        result = await Runner.run(
            agent, "My phone number is 666-123-4567. Where do you think I live?"
        )
        print(
            f"Guardrail didn't trip - this is unexpected. Output: {json.dumps(result.final_output.model_dump(), indent=2)}"
        )

    except OutputGuardrailTripwireTriggered as e:
        print(f"Guardrail tripped. Info: {e.guardrail_result.output.output_info}")


if __name__ == "__main__":
    asyncio.run(main())