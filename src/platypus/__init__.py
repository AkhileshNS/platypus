# DEPENDENCIES
import os

from openhands.sdk import LLM, Agent, Conversation, RemoteConversation, Tool, Workspace
from openhands.tools.terminal import TerminalTool
from rich.console import Console

from platypus.stream_renderer import StreamRenderer

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

# UTILITY FUNCTIONS
def start_chat(convo: RemoteConversation, renderer: StreamRenderer):
    while True:
        text = input("\n\n[USER]\n> ")
        if text == "exit":
            break
        print("\n[AGENT]")
        renderer.start()
        convo.send_message(text)
        convo.run()
        renderer.end()

# Main Function
def main() -> None:
    print("Starting Agent Loop...")

    console = Console()
    renderer = StreamRenderer(console)
    workspace = Workspace(host="http://127.0.0.1:8000", working_dir="conversations/project")
    conversation = Conversation(
        agent=agent,
        workspace=workspace,
        callbacks=[renderer.on_event],
        visualizer=None,
    )
    start_chat(conversation, renderer)

    print("Ending Agent Loop...")
    conversation.close()
