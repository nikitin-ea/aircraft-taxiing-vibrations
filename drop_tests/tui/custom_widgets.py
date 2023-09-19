import pyfiglet
from rich.text import Text
from textual.app import ComposeResult
from textual.widget import Widget
from textual.containers import Grid
from textual.widgets import Button, Static, Label, Input, Markdown

import visual_drop_test_constants as const

class AppLogo(Static):
    def compose(self) -> ComposeResult:        
        logo = Label(pyfiglet.figlet_format(const.APP_NAME,
                                            font='smslant'),
                    id="app-logo")
        info = Markdown(const.APP_INFO, id="app-info")
        yield logo 
        yield info


class InputWithAction(Widget):
    def __init__(self,
                 input_label: str,
                 button_label: str) -> None:
        self.label = Label(input_label)
        self.input = Input()
        self.button = Button(label=button_label)
        super().__init__()

    def compose(self) -> ComposeResult:  
        yield self.label
        yield self.input
        yield self.button


class LabeledInput(Widget):
    def __init__(self, input_label: str) -> None:
        self.label = Label(input_label)
        self.input = Input()
        super().__init__()

    def compose(self) -> ComposeResult:  
        yield self.label
        yield self.input
        