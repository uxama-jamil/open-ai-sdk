from agents import Agent,Runner,AsyncOpenAI,OpenAIChatCompletionsModel,set_tracing_disabled


externalProvider= AsyncOpenAI(
    api_key="AIzaSyB4BU0F5fGuXkkoWx5xhal-X9w81BZYcHw",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model=OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=externalProvider
)

set_tracing_disabled(True)

def runAgent():
    agent = Agent(name="Assistant", instructions="You are a helpful assistant",model=model)
    response = Runner.run_sync(starting_agent=agent,
                            input="What is the capital of blackHole?",
                            )
    print(response.final_output)





















