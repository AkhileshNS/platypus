# DEPENDENCIES
import os
import sys

from openhands.sdk import LLM, Agent, Conversation, Tool, LocalConversation
from openhands.tools.terminal import TerminalTool
from openhands.sdk.llm.streaming import ModelResponseStream

# CONSTANTS
llm = LLM(
    model=os.getenv("LLM_MODEL", "openrouter/deepseek/deepseek-v4.1-flash"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", None)
)

agent = Agent(
    llm=llm,
    tools=[
        Tool(name=TerminalTool.name)
    ],
)

# Utility Functions
def on_token(chunk: ModelResponseStream):
    """
    Process each streaming chunk as it arrives
    """
    choices = chunk.choices
    for choice in choices:
        delta = choice.delta
        if delta is not None:
            content = getattr(delta, "content", None)
            if isinstance(content, str):
                sys.stdout.write(content)
                sys.stdout.flush()

def start_chat(convo: LocalConversation):
    while True:
        text = input("Say something: ")
        if text == "exit":
            break
        convo.send_message(text)
        convo.run()

# Main Function
def main() -> None:
    print("Starting Agent Loop...")

    cwd = os.getcwd()
    conversation = Conversation(
        agent=agent,
        workspace=cwd,
        token_callbacks=[on_token]
    )
    start_chat(conversation)
    conversation.run()

    print("Ending Agent Loop...")
