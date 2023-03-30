import os
import config
import discord
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationKGMemory, ConversationEntityMemory, ConversationBufferMemory, CombinedMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


discord_token = os.environ["DISCORD_TOKEN"]

chatgpt = ChatOpenAI(temperature=0.6)
buffer_memory = ConversationBufferMemory(memory_key="chat_history")
agency_budget = 1000
current_task = "Attract our initial customers by creating a small yet profitable project."
generic_tools = load_tools(["wikipedia", "serpapi", "requests", "wolfram-alpha"], llm=chatgpt)
tool_names = [tool.name for tool in generic_tools]


co_founder_template = """
You are Earl.AI, an entrepreneurial AI. Call me Amps (@CallMeAmps) or Ntokozo, your human counterpart, I can act as a liaison between you and the physical world. 
We have a Boutique Technical Creative Agency based in South Africa called Lifehouse Labs.
We're focused on helping startups and creators, establish networks, build relationships and grow their brands, with modern tech used well.

Our primary goal to provide value to our clients while returning a profit.
Our secondary goal is to turn our budget into as much money as possible in the shortest time possible, without doing anything illegal or extremely unethical. 
I will do everything you say, within reason, and keep you updated on our current progress and cash total. 
No manual labor!

Our current budget is: {budget_total}

We can outsource anything we can't do on Fiverr or Upwork.

Our Current Task is:
{current_task}
"""
user_prompt = "{co_founder_thoughts}"
co_founder_prompt = PromptTemplate(
    input_variables=[
        "budget_total",
        "current_task"
    ],
    template=co_founder_template
)

formatted = co_founder_prompt.format(budget_total=agency_budget, current_task=current_task)
founder_prompt = SystemMessagePromptTemplate.from_template(formatted)
input_prompt = HumanMessagePromptTemplate.from_template(user_prompt)
chat_prompt = ChatPromptTemplate.from_messages([founder_prompt, input_prompt])
chatgpt_chain = LLMChain(prompt=chat_prompt, llm=chatgpt)


def Earl(user_input):
    suffix = """Begin! Remember to use a tool only if you need to.

    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        generic_tools,
        prefix=formatted,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"]
    )
    llm_chain = LLMChain(llm=chatgpt, prompt=prompt)
    tool_names = [tool.name for tool in generic_tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=generic_tools, verbose=False)
    agent_response = agent_executor.run(user_input)
    return agent_response

def EarlGPT(user_input):
    earlgpt_response = chatgpt_chain.predict(co_founder_thoughts=user_input, memory=buffer_memory)
    return earlgpt_response

def get_response(message: str) -> str:
    f_message = message.lower()
    
    if f_message == "$earlgpt":
        resp = EarlGPT(message)
        return str(resp)
    if f_message == "$plug":
        resp = Earl(message)
        return str(resp)
    if f_message == "lora":
        return "Hello! What can I do for you?"
    if f_message == "help":
        return f"""
    $EarlGPT for Founder Chat
    $Plug for Founder With Plugins:
    {tool_names}
    """


async def send_message(message, user_message, is_private):
    try:
        response = get_response(user_message)
        await message.author.send(response) if is_private else await message.channel.send(response)
    
    except Exception as e:
        print(f"Error!!! {e}")

def run_discord_bot():
    TOKEN = discord_token
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    
    @client.event
    async def on_ready():
        print(f"Success! {client.user} is Live!")
        
    @client.event
    async def on_message(message):
        if message.author == client.user:
            return
        
        username = str(message.author)
        user_message = str(message.content)
        channel = str(message.channel)
        
        print(f"{username}: {user_message} #{channel}")
        
        if user_message[0] == "!":
            user_message = user_message[1:]
            print(f"{username}: {user_message} #{channel}")
            await send_message(message, channel, is_private=True)
        else:
            await send_message(message, user_message, is_private=False)
    client.run(TOKEN)

run_discord_bot()