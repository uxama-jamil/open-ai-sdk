import asyncio
from typing import Any, Literal
from agents import (Agent,  GuardrailFunctionOutput, InputGuardrailTripwireTriggered, RunContextWrapper, 
                    Runner, AsyncOpenAI, OpenAIChatCompletionsModel, TResponseInputItem, Tool, ToolsToFinalOutputFunction, 
                     function_tool, input_guardrail, set_tracing_disabled)
from dotenv import load_dotenv
import os

from pydantic import BaseModel
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
This example shows how to use guardrails.

Guardrails are checks that run in parallel to the agent's execution.
They can be used to do things like:
- Check if input messages are off-topic
- Check that input messages don't violate any policies
- Take over control of the agent's execution if an unexpected input is detected

In this example, we'll setup an input guardrail that trips if the user input is containing any sensitive or abusive information.
If the guardrail trips, we'll respond with a refusal message.
"""


### 1. An agent-based guardrail that is triggered if the user input is containing any sensitive or abusive information
class SensitiveInput(BaseModel):
    reasoning: str
    is_sensitive_input: bool
    sensitive_words: list[str]


guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user input is containing any sensitive or abusive information.",
    output_type=SensitiveInput,
    model=model,
)


@input_guardrail
async def sensitive_input_guardrail(
    context: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """This is an input guardrail function, which happens to call an agent to check if the input
    is a sensitive or abusive input.
    """
    result = await Runner.run(guardrail_agent, input, context=context.context)
    final_output = result.final_output_as(SensitiveInput)
    print(f"sensitive_input_guardrail result is {final_output} ")
    return GuardrailFunctionOutput(
        output_info=final_output, #guardrail output if it trips
        tripwire_triggered=final_output.is_sensitive_input,
    )


### 2. The run loop


async def main():
    agent = Agent(
        name="Customer support agent",
        instructions="You are a customer support agent. You help customers with their questions.",
        input_guardrails=[sensitive_input_guardrail],
        model=model,
    )

    input_data: list[TResponseInputItem] = []

    while True:
        user_input = input("Enter a message: ")
        input_data.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        try:
            result = await Runner.run(agent, input_data)
            print(result.final_output)
            # If the guardrail didn't trigger, we use the result as the input for the next run
            input_data = result.to_input_list() #to_input_list() is used to convert the output to a list of input items containing input items and output items in the list
            print(f"input_data is {input_data} ")
        except InputGuardrailTripwireTriggered:
            # If the guardrail triggered, we instead add a refusal message to the input
            message = "Sorry, I can't help you b/c your input is containing sensitive or abusive information."
            print(message)
            input_data.append(
                {
                    "role": "assistant",
                    "content": message,
                }
            )

    # Sample run:
    # Enter a message: 
    # The capital of California is Sacramento.
    # Enter a message: Can you help me solve for x: 2x + 5 = 11
    # Sorry, I can't help you with your math homework.


if __name__ == "__main__":
    asyncio.run(main())