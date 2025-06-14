from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled,function_tool,handoff
import requests
import asyncio
from dataclasses import dataclass
from agents import RunContextWrapper

#guardrails
from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
)

set_tracing_disabled(True)
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



class MathHomeworkOutput(BaseModel):
    is_math_homework: bool
    reasoning: str

guardrail_agent = Agent( 
    name="Guardrail check",
    instructions="Check if the user is asking you to do their math homework.",
    model=model,
    output_type=MathHomeworkOutput,
)


@input_guardrail
async def math_guardrail( 
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    print(f"math_guardrail calling ... {input} and context is {ctx.context} ")
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    print(f"math_guardrail result is {result.final_output} and tripwire_triggered is {result.final_output.is_math_homework} ")

    return GuardrailFunctionOutput(
        output_info=result.final_output, 
        tripwire_triggered=result.final_output.is_math_homework,
    )


agent = Agent(  
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    model=model,
    input_guardrails=[math_guardrail],
)

async def main():
    # This should trip the guardrail
    try:
        response = await Runner.run(agent, "Hello, what is the weather in karachi?")
        print("--------------------------------")
        print(response.final_output)
        # print("Guardrail didn't trip - this is unexpected")

    except InputGuardrailTripwireTriggered as e:
        print("Math homework guardrail tripped")
        print(e)
        
asyncio.run(main())