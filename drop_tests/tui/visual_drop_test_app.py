# -*- coding: utf-8 -*-
import os
from enum import IntEnum

import numpy as np

from textual import work
from textual.worker import Worker, get_current_worker, WorkerState
from textual.app import ComposeResult, App
from textual.binding import Binding
from textual.widgets import Footer, Header, TabbedContent, TabPane, Button

from screens import Help, PopUpScreenChoice, PopUpScreenMessage
from tabs import Logger, PreProc, PostProc
from custom_widgets import AppLogo
import visual_drop_test_constants as const

from app_model import DropTestModel, State

class Table(IntEnum):
    T = 0
    U = 1
    V = 2
    DPHIDT = 3
    S = 4
    Y1 = 5
    Y = 6
    FX = 7
    FY = 8
    MZ = 9
    P = 10

class VirtualDropTestApp(App):
    """Represents a Virtual Drop Test Application.

    The `VirtualDropTestApp` class is responsible for creating and managing 
    the user interface of the application. It includes methods for composing 
    the UI, handling events, and performing actions.

    Attributes:
        SCREENS (dict): A dictionary mapping screen names to their corresponding 
        classes.
        BINDINGS (list): A list of key bindings for various actions in the app.
        CSS_PATH (str): The path to the CSS file for styling the app.

    Methods:
        compose(self) -> ComposeResult: Composes the user interface by yielding 
        child widgets.
        on_mount(self): Performs actions when the app is mounted.
        action_toggle_dark(self) -> None: Toggles the dark mode of the app.
        action_request_quit(self) -> None: Requests confirmation to quit 
        the app.

    """
    TITLE = const.APP_NAME
    SCREENS = {"help": Help}
    BINDINGS = [Binding("escape", "request_quit", "Выйти", 
                        key_display="Esc"),
                Binding("ctrl+d", "toggle_dark", "Сменить тему", 
                        key_display="ctrl+d"),
                Binding("ctrl+s", "make_screenshot", "Снимок экрана", 
                        key_display="ctrl+s"),
                Binding("question_mark", "push_screen('help')", "Помощь", 
                        key_display="?")]
    CSS_PATH = const.APP_CSS_PATH
    
    def compose(self) -> ComposeResult:
        """Composes the user interface of the app.

        Yields:
            ComposeResult: Child widgets for the app.
        """
        self.header = Header(name=const.APP_NAME, show_clock=True)
        self.app_logo = AppLogo()   
        self.logger = Logger()
        self.preproc = PreProc()
        self.postproc = PostProc()
        self.tabs = TabbedContent(initial="tab-preproc")
        
        self.header.tall = True
        yield self.header
        yield self.app_logo
        with self.tabs:
            with TabPane("Подготовка", id="tab-preproc"):
                yield self.preproc
            with TabPane("Результат", id="tab-postproc"):
                yield self.postproc
            with TabPane("Сообщения", id="tab0-log"):
                yield self.logger     
        yield Footer()
        
    def on_mount(self):
        """Performs actions when the app is mounted.
        """
        self.app_logo.styles.animate("opacity", value=1.0,  duration=2.0,
                                     easing="in_out_quart")
        self.tabs.styles.animate("opacity", value=1.0,  duration=2.0)
        self.logger.print_message(f"Программа {const.APP_NAME} запущена. "
                                  f"Добро пожаловать!")
    
    def action_toggle_dark(self) -> None:
        """Toggles the light/dark mode of the app.
        """
        self.dark = not self.dark
        theme = "темный" if self.dark else "светлый"
        self.logger.print_message(f"Интерфейс переключен в {theme} режим.")
        
    def action_make_screenshot(self) -> None:
        if not os.path.exists(const.SCREENSHOTS_PATH):
            os.makedirs(const.SCREENSHOTS_PATH)
        screenshot_fname = self.save_screenshot(filename=None,
                                                path=const.SCREENSHOTS_PATH)
        screenshot_fname = os.path.basename(screenshot_fname)
        self.logger.print_message(f"Снимок экрана {screenshot_fname} сохранен "
                                  f"в папку "
                                  f"{const.SCREENSHOTS_PATH}\{self.TITLE}")
      
    def action_request_quit(self) -> None:
        """Requests confirmation to quit the app.
        """
        def check_quit(quit: bool) -> None:
            if quit:
                self.exit()
        self.push_screen(PopUpScreenChoice("Вы действительно хотите выйти?"), 
                         check_quit)
        
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button == self.preproc.strut_strip.button:
            self.preproc.request_json_load(self.preproc.strut_strip, "стойки")
        elif event.button == self.preproc.tyre_strip.button:
            self.preproc.request_json_load(self.preproc.tyre_strip, "шины")
        elif event.button == self.preproc.check_button:
            self.preproc.show_parameters()
        elif event.button == self.preproc.start_button:
            self.check_and_run()
        elif event.button == self.postproc.first_strip.button:
            self.postproc.request_load_data()
        elif event.button == self.postproc.second_strip.button:
            self.postproc.request_save_data()

    def check_and_run(self):
        def check_clear(clear: bool) -> None:
            if clear:
                self.postproc.dt.clear()
                self.app.logger.print_message("Данные удалены!")
                self.run_analysis()
            else:
                self.push_screen(
                    PopUpScreenMessage('Анализ не запущен!')
                    )

        if self.postproc.dt.row_count > 1:
            self.push_screen(
                PopUpScreenChoice('Существующие результаты будут удалены!'), 
                check_clear
            )
        else:
            self.run_analysis()
              
    @work(exclusive=True, thread=True)      
    def run_analysis(self):
        self.logger.print_message("Начало виртуальных испытаний...")
        self.preproc.create_param_dict()
        if self.preproc.param_dict is None or self.preproc.param_dict == {}:
            self.logger.print_message("Параметры моделей не заданы!")
            self.logger.print_message(self.preproc.param_dict)
            return
        
        worker = get_current_worker()
        
        self.preproc.start_button.disabled = True
    
        model = DropTestModel(self.preproc.param_dict, 
                              self.logger,
                              self.preproc.progress_bar)
        self.logger.print_message("Модель опоры шасси создана.")
        
        try:
            result = model.get_result()
        except Exception as exc:
            self.logger.print_message(f"{exc}: ошибка интегрирования.")
            return
        finally:
            self.logger.print_message("Завершение анализа...")
            self.preproc.start_button.disabled = False
            
        data = self.map_result_to_data(result)
        self.logger.print_message("Распечатка результатов в таблицу...")
        if not worker.is_cancelled:
            self.call_from_thread(self.postproc.fill_data_table, data)
            self.preproc.start_button.remove_class("idle")
                
    def map_result_to_data(self, result):
        if result is None:
            return
        
        num_points = result.t.shape[0]
        data = np.zeros((11, num_points)) 
        data[Table.T] = result.t
        data[Table.U] = result.y[State.U]
        data[Table.V] = result.y[State.V]
        data[Table.DPHIDT] = result.y[State.DPHIDT]
        data[Table.S] = result.y[State.S]
        data[Table.Y1] = result.axle_position     
        data[Table.Y] = result.y[State.Y]
        data[Table.FX] = result.horizontal_force
        data[Table.FY] = result.vertical_force
        data[Table.MZ] = result.braking_torque
        data[Table.P] = result.pressure
        return data.T

if __name__ == "__main__":
    app = VirtualDropTestApp()
    app.run()
    