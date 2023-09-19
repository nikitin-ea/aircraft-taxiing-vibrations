from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Grid, Container
from textual.screen import Screen, ModalScreen
from textual.widgets import (Footer, Button, Label, Markdown, Pretty)

import visual_drop_test_constants as const
 
class Help(Screen):
    BINDINGS = [("escape,space,q,question_mark", "pop_screen", "Закрыть")]

    def compose(self) -> ComposeResult:
        yield Markdown(Path(const.HELP_FILE).read_text(encoding="utf-8"))
        yield Footer()
        
class ParametersView(Screen):
    BINDINGS = [("escape,space,q,question_mark", "pop_screen", "Закрыть")]
    
    def __init__(self, dictionary):
        self.dictionary = dictionary
        super().__init__()
    
    def compose(self) -> ComposeResult:
        yield Label("Параметры модели", id="param-screen-header")
        yield Pretty(self.dictionary)
        yield Footer()
        
class PopUpScreenMessage(ModalScreen[bool]):
    def __init__(self, message: str) -> None:
        self.label = Label(message, id="question")
        self.ok_button = Button("Да",
                                variant="error",
                                id="ok",
                                classes="msg-button")
        super().__init__()
        
    def compose(self) -> ComposeResult:
        yield Container(self.label,
                self.ok_button,
                id="dialog")
        
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            self.dismiss(True)
        else:
            self.dismiss(False)


class PopUpScreenChoice(ModalScreen[bool]):
    def __init__(self, message: str) -> None:
        self.label = Label(message, id="question")
        self.ok_button = Button("Да",
                                variant="error",
                                id="ok",
                                classes="msg-button-choice")
        self.cancel_button = Button("Отмена",
                                    variant="primary", 
                                    id="cancel", 
                                    classes="msg-button-choice")
        super().__init__()
        
    def compose(self) -> ComposeResult:
        yield Grid(self.label,
                   self.ok_button,
                   self.cancel_button,
                   id="dialog")
        
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            self.dismiss(True)
        else:
            self.dismiss(False)
