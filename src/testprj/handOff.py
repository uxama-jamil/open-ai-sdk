from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled,function_tool,handoff
import requests
import asyncio

set_tracing_disabled(True)
# Set up the external language model provider (Google's Gemini, using OpenAI-compatible API)
externalProvider = AsyncOpenAI(
    api_key="AIzaSyB4BU0F5fGuXkkoWx5xhal-X9w81BZYcHw",  # API key for Gemini
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # Gemini-compatible endpoint
)

# Create a chat completions model using the Gemini model via the OpenAI API interface
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",  # Name of the Gemini model to use
    openai_client=externalProvider  # The provider configured above
)

history_tutor_agent = Agent(
    name="History Tutor",
    instructions="You are a history tutor. You are given a question and you need to answer it based on the history.",
    model=model,
    handoff_description = " This is used to get the historical context of the question"
)

math_tutor_agent = Agent(
    name="Math Tutor",
    instructions="You are a math tutor. You are given a question and you need to answer it based on the math.",
    model=model,
    handoff_description = "This is used to get the math context of the question"
)

astronaunt_tutor_agent = Agent(
    name="Astronaut Tutor",
    instructions="You are a astronaut tutor. You are given a question and you need to answer it based on the astronaut.",
    model=model,
    handoff_description = "This is used to get the astronaut context of the question"
)

science_tutor_agent = Agent(
    name="Science Tutor",
    instructions="You are a science tutor. You are given a question and you need to answer it based on the science.",
    model=model,
    handoff_description = "This is used to get the science context of the question"
)

sindhi_tutor_agent = Agent(
    name="Sindhi Tutor",
    instructions="You are a sindhi tutor. You are given a question and you need to answer it based on the sindhi.",
    model=model,
    handoff_description = "This is used to get the sindhi context of the question"
)

handoff_history_tutor = handoff(
    agent=history_tutor_agent,
    tool_name_override = "custom_history_tutor",
    tool_description_override = "This tool is used to get the historical context of the question",
    on_handoff = lambda x: print(f"Handing off to history tutor: {x}")
)

handoff_math_tutor = handoff(
    agent=math_tutor_agent,
    tool_name_override = "custom_math_tutor",
    tool_description_override = "This tool is used to get the math context of the question",
    on_handoff = lambda x: print(f"Handing off to math tutor: {x}")
)

handoff_astronaunt_tutor = handoff(
    agent=astronaunt_tutor_agent,
    # tool_name_override = "custom_astronaunt_tutor",
    # tool_description_override = "This tool is used to get the astronaut context of the question",
    on_handoff = lambda x: print(f"Handing off to astronaut tutor: {x}")
)

handoff_science_tutor = handoff(
    agent=science_tutor_agent,
    on_handoff = lambda x: print(f"Handing off to science tutor: {x}")
)

handoff_sindhi_tutor = handoff(
    agent=sindhi_tutor_agent,
    on_handoff = lambda x: print(f"Handing off to sindhi tutor: {x}")
)

# agent = Agent(
#     name="Main Agent",
#     instructions="You are a main agent. You are given a question and you need to answer it based on the history or math.",
#     model=model,
#     handoffs = [handoff_math_tutor,handoff_history_tutor]
# )

agent_f_1 = Agent(
    name="Main Agent",
    instructions="You are a main agent. You are given a question and you need to answer it based on the history or math.",
    model=model,
    handoffs = [history_tutor_agent,math_tutor_agent]
)

agent_f_2 = Agent(
    name="Main Agent",
    instructions="You are a main agent. You are given a question and you need to answer it based on the history or math.",
    model=model,
    handoffs = [astronaunt_tutor_agent,science_tutor_agent,sindhi_tutor_agent]
)

response = asyncio.run(Runner.run(
    starting_agent=agent_f_1,
    input="what is the integral of 1/x and who is the founder of india?"
))

response_f_2 = asyncio.run(Runner.run(
    starting_agent=agent_f_2,
    input="what is the diameter of jupyter and sun,this is the question of astronaut?"
))

print(response.final_output)
print(response.last_agent.name)
print(response_f_2.final_output)
print(response_f_2.last_agent.name)


