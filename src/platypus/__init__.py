# DEPENDENCIES
import os
import sys

from openhands.sdk import LLM, Agent, Conversation, Tool, LocalConversation
from openhands.tools.terminal import TerminalTool
from openhands.sdk.conversation import ConversationVisualizerBase
from openhands.sdk.event import ActionEvent, ObservationEvent, AgentErrorEvent, Event
from openhands.sdk.llm import content_to_str

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text
from rich.panel import Panel

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

console = Console()
BG = "on grey19"          # dark gray background, shared by both
TOOL_FG = "bright_cyan"   # tool call colour
OUT_FG = "green"          # tool output colour

# Utility Functions
class ToolLogger(ConversationVisualizerBase):
    def on_event(self, event: Event) -> None:
        if isinstance(event, ActionEvent):
            body = Text()
            body.append(event.tool_name, style=f"bold {TOOL_FG}")
            if event.action is not None:
                body.append(f"\n{event.action.model_dump()}", style=TOOL_FG)
            console.print(Panel(body, title="tool call", title_align="left", border_style=TOOL_FG, style=BG))
        elif isinstance(event, ObservationEvent):
            body = Text("".join(content_to_str(event.observation.to_llm_content)), style=OUT_FG)
            console.print(Panel(body, title="output", title_align="left", border_style=OUT_FG, style=BG))
        elif isinstance(event, AgentErrorEvent):
            console.print(Panel(Text(event.error, style="red"), title="error", title_align="left", border_style="red", style=BG))


class StreamRenderer:
    def __init__(self) -> None:
        self.console = console
        self.buffer: list[str] = []
        self.live: Live | None = None

    def start(self) -> None:
        self.buffer.clear()
        self.live = Live(
            console=self.console,
            refresh_per_second=8,
            vertical_overflow="visible"
        )
        self.live.start()

    def end(self) -> None:
        if self.live is None:
            return
        self.live.update(Markdown("".join(self.buffer)))
        self.live.stop()
        self.live = None

    def __call__(self, chunk) -> None:
        for choice in chunk.choices:
            delta = choice.delta
            if delta and isinstance(delta.content, str):
                self.buffer.append(delta.content)
        if self.live is not None:
            self.live.update(Text("".join(self.buffer)))


def start_chat(convo: LocalConversation, renderer: StreamRenderer):
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

    cwd = os.getcwd()
    renderer = StreamRenderer()
    conversation = Conversation(
        agent=agent,
        workspace=cwd,
        token_callbacks=[renderer],
        visualizer=ToolLogger,
    )
    start_chat(conversation, renderer)

    print("Ending Agent Loop...")
