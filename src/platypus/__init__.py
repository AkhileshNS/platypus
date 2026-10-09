# DEPENDENCIES
import os
import sys

from openhands.sdk import LLM, Agent, Conversation, Tool, LocalConversation
from openhands.tools.terminal import TerminalTool
from openhands.sdk.llm.streaming import ModelResponseStream
from openhands.sdk.conversation import ConversationVisualizerBase
from openhands.sdk.event import ActionEvent, ObservationEvent, AgentErrorEvent, Event

# CONSTANTS
llm = LLM(
    model=os.getenv("LLM_MODEL", "openrouter/deepseek/deepseek-v4.1-flash"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", None),
    stream=True
)

agent = Agent(
    llm=llm,
    tools=[
        Tool(name=TerminalTool.name)
    ],
)

# Utility Functions
class ToolLogger(ConversationVisualizerBase):
    def on_event(self, event: Event) -> None:
        if isinstance(event, (ActionEvent, ObservationEvent, AgentErrorEvent)):
            print(f"\n{event.visualize}")

def on_token(chunk: ModelResponseStream) -> None:
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
        text = input("\n\n[USER]\n> ")
        print("\n[AGENT]")
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
        token_callbacks=[on_token],
        visualizer=ToolLogger,
    )
    start_chat(conversation)

    print("Ending Agent Loop...")
