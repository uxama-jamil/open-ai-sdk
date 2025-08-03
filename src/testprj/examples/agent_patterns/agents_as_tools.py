# Import core components from the openai-agents framework
import asyncio
from agents import Agent, AgentHooks, ModelSettings, RunContextWrapper, RunHooks, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, Tool, function_tool, set_tracing_disabled
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
    model="gemini-1.5-flash",  # Name of the Gemini model to use
    openai_client=externalProvider  # The provider configured above
)

# Disable internal tracing/logging (optional – for performance or privacy)
set_tracing_disabled(True)


"""
    This example shows the agents-as-tools pattern. The frontline agent receives a user message and
    then picks which agents to call, as tools. In this case, it picks from a set of translation
    agents.
"""

class CustomAgentHook(AgentHooks):
    async def on_start(self, context: RunContextWrapper, agent: Agent):
        print(f"Agent started {agent.name} ")
    async def on_end(self, context: RunContextWrapper, agent: Agent, output: any):
        print(f"Agent ended {agent.name} ")
    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool):
        print(f"Agent Tool started {tool.name} ")
    async def on_tool_end(self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str):
        print(f"Agent Tool ended {tool.name} ")
    async def on_handoff(self, context: RunContextWrapper, agent: Agent, source:Agent):
        print(f"Agent Handoff {agent.name} to {source.name} ")


"""
    name : name of the agent
    Instructions : what the agent should do
    handoff_description: it is only used when You are turning an Agent into a tool using .as_tool()
    model: (Gemini) 
"""
spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish",
    handoff_description="An english to spanish translator",
    model=model,

)

french_agent = Agent(
    name="french_agent",
    instructions="You translate the user's message to French",
    handoff_description="An english to french translator",
    model=model,

)

italian_agent = Agent(
    name="italian_agent",
    instructions="You translate the user's message to Italian",
    handoff_description="An english to italian translator",
    model=model,

)

urdu_agent = Agent(
    name="urdu_agent",
    instructions="You translate the user's message to Urdu",
    handoff_description="An english to urdu translator",
    model=model,

)

"""
This is the main agent (also called a tool-using agent):
    It does not translate itself.
    It looks at your input and decides which of the mini-agents to use.
    It has access to the 3 translation agents as tools via .as_tool(...).
"""
orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate."
        "If asked for multiple translations, you call the relevant tools in order."
        "You never translate on your own, you always use the provided tools."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
        italian_agent.as_tool(
            tool_name="translate_to_italian",
            tool_description="Translate the user's message to Italian",
        ),
        urdu_agent.as_tool(
            tool_name="translate_to_urdu",
            tool_description="Translate the user's message to Urdu",
        ),
    ],
    model=model,
    hooks=CustomAgentHook(),
)

# synthesizer_agent = Agent(
#     name="synthesizer_agent",
#     instructions="You inspect translations, correct them if needed, and produce a final concatenated response.",
#     model=model,
# )


def main():
    msg = input("Hi! What would you like translated, and to which languages? ")

    orchestrator_result = asyncio.run(Runner.run(orchestrator_agent, msg))
    print(f"Orchestrator result: {orchestrator_result.final_output}")
    # Run the entire orchestration in a single trace
    # with trace("Orchestrator evaluator"):

        # for item in orchestrator_result.new_items:
        #     if isinstance(item, MessageOutputItem):
        #         text = ItemHelpers.text_message_output(item)
        #         if text:
        #             print(f"  - Translation step: {text}")

        # synthesizer_result = await Runner.run(
        #     synthesizer_agent, orchestrator_result.to_input_list()
        # )

    # print(f"\n\nFinal response:\n{synthesizer_result.final_output}")


if __name__ == "__main__":
    main()