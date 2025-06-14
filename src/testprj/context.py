from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled,function_tool,handoff
import requests
import asyncio
from dataclasses import dataclass
from agents import RunContextWrapper

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



# -----------------------------!st way of providing context to agent using tools-----------------------------

# dataclass is used to create a class that is used to store the user information
@dataclass
class UserInfo:
    name: str
    champak:bool
    age: int

@function_tool
def fetch_user_details(user_info: RunContextWrapper):
    '''
    This function is used to fetch the user details like age, name and champak
    @param user_info: RunContextWrapper 
    name: str
    champak: bool
    age: int
    @return: str
    '''
    # print(f"fetch_user_details calling ...  {wrapper.context.name} is {wrapper.context.age} years old and he is {wrapper.context.champak} champak ")
    return f"{user_info.context.name} is {user_info.context.age} years old and he is {user_info.context.champak} champak"

# 
async def  runAgentAsync ():
    user_info = UserInfo(name="Usama",age=47,champak=True)
    # Create an agent with a name and some instructions
    print(f"running agent")
    agent = Agent[UserInfo](
        name="Assistant",
        instructions="You are a helpful assistant",
        model=model,  # Uses the Gemini model set above
        tools=[fetch_user_details] # This is the tool that will be used to get the temperature of the city
    )

    # Run the agent synchronously with a given input
    response = await Runner.run(
        context=user_info,
        starting_agent=agent,
        input="what is the name and age of the user and is he mad?"  # Input message to the assistant
    )

    # Print the agent's final response
    print(response.final_output)
    
# -----------------------------!st way of providing context to agent using tools-----------------------------


# -----------------------------2nd way of providing context to agent using main agent instructions(system prompt)-----------------------------
async def  mainAgentInstructions ():
    user_info = UserInfo(name="Usama",age=47,champak=True)
    # Create an agent with a name and some instructions
    print(f"running agent")
    agent = Agent[UserInfo](
        name="Assistant",
        instructions=f"""You are a helpful assistant. The user you're assisting is {user_info.name}, 
        {user_info.age} years old and he is {user_info.champak} champak.
        You need to use the tool to get the user details.
        """,
        model=model,  # Uses the Gemini model set above
    )
    response = await Runner.run(
        context=user_info,
        starting_agent=agent,
        input="what is the name and age of the user and is he mad?"  # Input message to the assistant
    )

    # Print the agent's final response
    print(response.final_output)

# -----------------------------2nd way of providing context to agent using main agent instructions(system prompt)-----------------------------

# -----------------------------3rd way of providing context to in user input promt -----------------------------

async def  userInputPromt ():
    user_info = UserInfo(name="Moiz",age=47,champak=True)
    # Create an agent with a name and some instructions
    print(f"running agent")
    agent = Agent[UserInfo](
        name="Assistant",
        instructions=f"""You are a helpful assistant.
        """,
        model=model,  # Uses the Gemini model set above
    )
    response = await Runner.run(
        context=user_info,
        starting_agent=agent,
        input=f"""Context:
        The user you're assisting is {user_info.name}, {user_info.age} years old and he is {user_info.champak} champak.
        what is the name and age of the user and is he mad?
        """  # Input message to the assistant
    )

    # Print the agent's final response
    print(response.final_output)
    
# -----------------------------3rd way of providing context to in user input promt -----------------------------

# -----------------------------4th way of providing context to using RAG -----------------------------
# -----------------------------4th way of providing context to using RAG -----------------------------



asyncio.run(userInputPromt())




    