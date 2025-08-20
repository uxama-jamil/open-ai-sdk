import asyncio
from typing import Any, Literal
import uuid
from agents import (Agent, AgentHooks, FunctionToolResult, ModelSettings, RawResponsesStreamEvent, RunContextWrapper, RunHooks, 
                    Runner, AsyncOpenAI, OpenAIChatCompletionsModel, TResponseInputItem, Tool, ToolsToFinalOutputFunction, 
                    ToolsToFinalOutputResult, function_tool, set_tracing_disabled)
from dotenv import load_dotenv
import os
from .utils import image_to_base64,get_available_models
from pydantic import BaseModel
load_dotenv()

# Set up the external language model provider (Ollama)
externalProvider = AsyncOpenAI(
    api_key="ollama",  # dummy value to satisfy SDK
    base_url="http://localhost:11434/v1"  # local Ollama endpoint
)

# Create a chat completions model using the Llama vision model via the OpenAI API interface
model = OpenAIChatCompletionsModel(
    model="llama3.2-vision",  # Use the vision-specific model
    openai_client=externalProvider
)




# Disable internal tracing/logging (optional – for performance or privacy)
set_tracing_disabled(True)

FILEPATH = os.path.join(os.path.dirname(__file__), "media/image.jpg")





async def main():
    # Print base64-encoded image
    b64_image = image_to_base64(FILEPATH)

    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant that can analyze images. Describe what you see in detail.",
        model=model,
    )

    # Try the OpenAI-compatible format first
    result = await Runner.run(
        agent,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}",
                            "detail": "high"
                        }
                    }
                ],
            }
        ],
    )
    print(result.final_output)


# Alternative approach if the above doesn't work
async def main_alternative():
    b64_image = image_to_base64(FILEPATH)

    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant that can analyze images. Describe what you see in detail.",
        model=model,
    )

    # Alternative format - some Ollama setups expect this
    result = await Runner.run(
        agent,
        [
            {
                "role": "user",
                "content": f"What do you see in this image? [Image data: data:image/jpeg;base64,{b64_image}]",
            }
        ],
    )
    print(result.final_output)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"First approach failed: {e}")
        print("Trying alternative approach...")
        asyncio.run(main_alternative())