from openhands.sdk.event import (
    ActionEvent,
    AgentErrorEvent,
    Event,
    ObservationEvent,
    StreamingDeltaEvent,
)
from openhands.sdk.llm import content_to_str
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

BG = "on grey19"
TOOL_FG = "bright_cyan"
OUT_FG = "green"


class StreamRenderer:
    """Single owner of the terminal: tool panels + streamed text in one Live region."""

    def __init__(self, console: Console):
        self.console = console
        self.panels: list[Panel] = []
        self.buffer: list[str] = []
        self.live: Live | None = None

    def start(self) -> None:
        self.panels.clear()
        self.buffer.clear()
        self.live = Live(console=self.console, refresh_per_second=8)
        self.live.start()

    def end(self) -> None:
        if self.live is None:
            return
        self.live.update(self._render(final=True))
        self.live.stop()
        self.live = None

    def on_event(self, event: Event) -> None:
        if isinstance(event, StreamingDeltaEvent):
            if event.content:
                self.buffer.append(event.content)
        elif isinstance(event, ActionEvent):
            self.panels.append(self._tool_panel(event))
        elif isinstance(event, ObservationEvent):
            self.panels.append(self._output_panel(event))
        elif isinstance(event, AgentErrorEvent):
            self.panels.append(
                Panel(Text(event.error, style="red"), title="error",
                      title_align="left", border_style="red", style=BG)
            )
        else:
            return
        if self.live is not None:
            self.live.update(self._render())

    def _render(self, *, final: bool = False):
        text = "".join(self.buffer)
        body = Markdown(text) if final else Text(text)
        return Group(*self.panels, body)

    def _tool_panel(self, event: ActionEvent) -> Panel:
        body = Text()
        body.append(event.tool_name, style=f"bold {TOOL_FG}")
        if event.action is not None:
            body.append(f"\n{event.action.model_dump()}", style=TOOL_FG)
        return Panel(body, title="tool call", title_align="left",
                     border_style=TOOL_FG, style=BG)

    def _output_panel(self, event: ObservationEvent) -> Panel:
        body = Text("".join(content_to_str(event.observation.to_llm_content)),
                    style=OUT_FG)
        return Panel(body, title="output", title_align="left",
                     border_style=OUT_FG, style=BG)
