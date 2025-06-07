from agents import Agent,Runner,AsyncOpenAI,OpenAIChatCompletionsModel,set_tracing_disabled,function_tool


externalProvider= AsyncOpenAI(
    api_key="AIzaSyB4BU0F5fGuXkkoWx5xhal-X9w81BZYcHw",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model=OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=externalProvider
)

#This is a tool that returns the temperature of the city
@function_tool
def get_temperature(city:str):
    """
    Retrieve the current temperature for a given city.

    Parameters:
        city (str): The name of the city for which to retrieve the temperature.

    Returns:
        str: A string describing the temperature in the specified city.
    
    Example:
        >>> get_temperature("Paris")
        'The temperature in Paris is 20 degrees Celsius'
    """
    return f"The temperature in {city} is 20 degrees Celsius"

set_tracing_disabled(True)

def runAgent():
    agent = Agent(name="Assistant", instructions="You are a helpful assistant",model=model,tools=[get_temperature])
    response = Runner.run_sync(starting_agent=agent,
                            input="What is the temperature in the karachi?",
                            )
    print(response.final_output)
