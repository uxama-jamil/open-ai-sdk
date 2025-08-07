import asyncio
from dataclasses import dataclass
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
This example shows the LLM as a judge pattern. The first agent generates an outline for a story.
The second agent judges the outline and provides feedback. We loop until the judge is satisfied
with the outline.
"""

story_outline_generator = Agent(
    name="story_outline_generator",
    instructions=(
        "You generate a very short story outline based on the user's input. "
        "If there is any feedback provided, use it to improve the outline."
    ),
    model=model,
)


@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]


evaluator = Agent[None](
    name="evaluator",
    instructions=(
        "You evaluate a story outline and decide if it's good enough. "
        "If it's not good enough, you provide feedback on what needs to be improved. "
        "Never give it a pass on the first try. After 5 attempts, you can give it a pass if the story outline is good enough - do not go for perfection"
    ),
    output_type=EvaluationFeedback,
    model=model,
)


async def main() -> None:
    msg = input("What kind of story would you like to hear? ")
    input_items: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    latest_outline: str | None = None

    # We'll run the entire workflow in a single trace
    
    while True:
        story_outline_result = await Runner.run(
            story_outline_generator,
            input_items,
        )
        
        """
        4️⃣ 1. input_items = story_outline_result.to_input_list()
        Purpose:
        to_input_list() converts the entire conversation history (including the latest agent response) 
        into a format that can be fed to the next agent in the pipeline.
        Why We Need It:
        In a multi-agent conversation, each agent needs to see the full context of what happened before. 
        This method ensures conversation continuity.
        example: The result contains both input and output
        story_outline_result = {
            "input_history": [{"content": "Write a story about a dragon", "role": "user"}],
            "new_response": [{"content": "Here's a story outline: A young dragon learns to fly...", "role": "assistant"}]
        }

        # to_input_list() combines everything into conversation format
        input_items = story_outline_result.to_input_list()
        # Result:
        input_items = [
            {"content": "Write a story about a dragon", "role": "user"},
            {"content": "Here's a story outline: A young dragon learns to fly...", "role": "assistant"}
        ]
        """

        input_items = story_outline_result.to_input_list()
        """
        2️⃣ Purpose of new_items
        new_items extracts only the latest responses from the agent (not the entire conversation history).
         Example output:
         story_outline_result.new_items = [
            {
                "role": "assistant",
                "content": "In the year 3000, AI robots rebel against their creators..."
            }
        ]
        💡 Note: This does NOT include the user's original input — just the agent's new reply
        """
        new_responses = story_outline_result.new_items
        
        """
        Purpose of ItemHelpers.text_message_outputs()
        This helper function converts complex agent response objects into simple, readable text strings.
        
        #example
                # story_outline_result.new_items contains structured data like this:
        new_items = [
            {
                "type": "message",
                "role": "assistant", 
                "content": [
                    {
                        "type": "text",
                        "text": "Here's your story outline:\n\n1. Introduction: A young dragon named Ember lives in a mountain cave..."
                    }
                ],
                "metadata": {...},
                "timestamp": "2024-01-15T10:30:00Z"
            }
        ]

        # ItemHelpers.text_message_outputs() extracts just the text content
        text_only_response = ItemHelpers.text_message_outputs(new_items)
        # Result: "Here's your story outline:\n\n1. Introduction: A young dragon named Ember lives in a mountain cave..."
        """
        text_only_response = ItemHelpers.text_message_outputs(new_responses)
        latest_outline = text_only_response
        print("Story outline generated")

        evaluator_result = await Runner.run(evaluator, input_items)
        result: EvaluationFeedback = evaluator_result.final_output

        print(f"Evaluator score: {result.score}")

        if result.score == "pass":
            print("Story outline is good enough, exiting.")
            break

        print("Re-running with feedback")

        input_items.append({"content": f"Feedback: {result.feedback}", "role": "user"})

    print(f"Final story outline: {latest_outline}")


if __name__ == "__main__":
    asyncio.run(main())