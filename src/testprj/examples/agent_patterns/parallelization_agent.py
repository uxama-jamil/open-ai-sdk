import asyncio
import json
from typing import Any, Literal
from agents import (Agent,  GuardrailFunctionOutput, InputGuardrailTripwireTriggered, ItemHelpers, OutputGuardrailTripwireTriggered, RunContextWrapper, 
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
This example shows the parallelization pattern. We run the agent three times in parallel, and pick
the best result.
"""

spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish",
    model=model,
)

translation_picker = Agent(
    name="translation_picker",
    instructions="You pick the best Spanish translation from the given options.",
    model=model,
)


async def main():
    msg = input("Hi! Enter a message, and we'll translate it to Spanish.\n\n")

    """
    What it does:

        Runs the same translation agent 3 times in parallel
        asyncio.gather() waits for all three to complete
        Each run might produce slightly different translations due to LLM randomness

        Example output:
        Translation 1: "Hola, ¿cómo estás?"
        Translation 2: "Hola, ¿qué tal?"
        Translation 3: "Saludos, ¿cómo te encuentras?"
    """
    res_1, res_2, res_3 = await asyncio.gather(
        Runner.run(
            spanish_agent,
            msg,
        ),
        Runner.run(
            spanish_agent,
            msg,
        ),
        Runner.run(
            spanish_agent,
            msg,
        ),
    )
    #extracts just the text content from the latest response of all three
    outputs = [
        ItemHelpers.text_message_outputs(res_1.new_items),
        ItemHelpers.text_message_outputs(res_2.new_items),
        ItemHelpers.text_message_outputs(res_3.new_items),
    ]
    #join the outputs into a single string
    translations = "\n\n".join(outputs)
    print(f"\n\nTranslations:\n\n{translations}")

    best_translation = await Runner.run(
        translation_picker,
        f"Input: {msg}\n\nTranslations:\n{translations}",
    )

    print("\n\n-----")

    print(f"Best translation: {best_translation.final_output}")


if __name__ == "__main__":
    asyncio.run(main())