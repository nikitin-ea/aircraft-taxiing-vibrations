import os
import json
from datetime import datetime

import numpy as np
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Container
from textual.widgets import RichLog, ProgressBar, Button, DataTable

from custom_widgets import InputWithAction, LabeledInput
from screens import PopUpScreenMessage, PopUpScreenChoice, ParametersView

class PreProc(ScrollableContainer):
    """PreProc class.

    A class that represents a pre-processing container for a specific task. 
    It contains various input parameters and buttons for analysis and testing.

    Attributes:
        strut_strip (InputWithAction): An input field with an associated button 
        for loading strut parameters.
        tyre_strip (InputWithAction): An input field with an associated button 
        for loading tyre parameters.
        mass_input (LabeledInput): An input field for specifying the mass of the 
        cargo.
        energy_input (LabeledInput): An input field for specifying the impact 
        energy.
        spinup_input (LabeledInput): An input field for specifying the spin-up 
        speed.
        angle_input (LabeledInput): An input field for specifying the 
        installation angle.
        time_input (LabeledInput): An input field for specifying the integration 
        time.
        points_input (LabeledInput): An input field for specifying the number of 
        points.
        check_button (Button): A button for checking the parameters.
        start_button (Button): A button for starting the analysis.
        param_dict (dict): A dictionary to store the parameters.
        strut_dict (dict): A dictionary to store the strut parameters.
        tyre_dict (dict): A dictionary to store the tyre parameters.

    Methods:
        compose() -> ComposeResult: Composes and yields containers for the strut 
        parameters, test parameters, and analysis.
        on_button_pressed(event: Button.Pressed) -> None: Handles button press 
        events and performs corresponding actions.
        request_json_load(input, which) -> dict: Requests loading of JSON data 
        from a file and returns the loaded data.
        show_parameters() -> None: Shows the parameters if both strut and tyre 
        parameters are loaded.
        run_analysis() -> None: Runs the analysis.
    """
    def __init__(self):
        super().__init__()
        self.strut_strip = InputWithAction(input_label="Параметры амортизатора",
                                        button_label="Открыть")
        self.tyre_strip = InputWithAction(input_label="Параметры шины",
                                         button_label="Открыть")
        self.mass_input = LabeledInput(input_label="Масса груза, т")
        self.energy_input = LabeledInput(input_label="Энергия удара, кДж")
        self.spinup_input = LabeledInput(input_label="Скорость раскрутки, км/ч")
        self.angle_input = LabeledInput(input_label="Угол установки, °")
        self.time_input = LabeledInput(input_label="Время интегрирования, с")
        self.points_input = LabeledInput(input_label="Количество точек")
        
        self.progress_bar = ProgressBar(total=100, 
                                        id="progress-bar")
        
        self.check_button = Button(label="Проверить", id="button-checker")
        self.start_button = Button(label="Испытать", id="button-analysis")
        
        self.param_dict = {}
        self.strut_dict = {}
        self.tyre_dict = {}
        self.test_dict = {}
        
    def compose(self) -> ComposeResult:
        cnt_models =  Container(self.strut_strip,
                                self.tyre_strip,
                        id="container-load-parameters")
        cnt_models.border_title = "Параметры опоры шасси"
        cnt_models.styles.border_title_align = "center"
        yield cnt_models
        
        cnt_test = Container(self.mass_input ,
                             self.energy_input,
                             self.time_input,
                             self.angle_input,
                             self.spinup_input,
                             self.points_input,
                             id="container-test-parameters")
        cnt_test.border_title="Параметры виртуальных испытаний"
        cnt_test.styles.border_title_align = "center"
        yield cnt_test

        cnt_analysis = Container(Container(self.progress_bar,
                                           id="subcontainer-progressbar"),
                                 Container(self.check_button,
                                           self.start_button,
                                           id="subcontainer-analysis"),
                                 id="container-analysis")
        cnt_analysis.border_title="Анализ"
        cnt_analysis.styles.border_title_align = "center"
        yield cnt_analysis
            
    @staticmethod        
    def change_button_color(button, success) -> None:
        if success:
            button.remove_class("failed")
            button.add_class("succeeded")
        else:
            button.remove_class("succeeded")
            button.add_class("failed")
            
    def request_json_load(self, strip, which):
        fname = strip.input.value.strip(''' '"''')
        
        if fname == "" or fname is None:
            self.change_button_color(strip.button, False)
            return
        
        try:
            with open(fname, "r") as file:
                json_data = json.load(file)
        except Exception as exc:
            self.app.logger.print_message(f"{exc}: Ошибка при открытии файла "
                                          f"{os.path.basename(fname)}!")
            strip.input.value = ""
            self.change_button_color(strip.button, False)
            return
        
        json_data["filename"] = fname
        self.app.logger.print_message(f"Файл {os.path.basename(fname)} "
                                      f"успешно открыт. "
                                      f"Параметры {which} загружены.")
        self.change_button_color(strip.button, True)
        
        if which == "стойки":
            self.strut_dict = json_data
        elif which == "шины":
            self.tyre_dict = json_data
          
    def collect_input_data(self):
        try:
            self.test_dict = {
                "impact-energy-kJ": float(self.energy_input.input.value),
                "cage-mass-t": float(self.mass_input.input.value),
                "spinup-kmh": float(self.spinup_input.input.value),
                "setup-angle": float(self.angle_input.input.value),
                "termination-s": float(self.time_input.input.value),
                "num-points": int(self.points_input.input.value),
            }
        except Exception as exc:
            self.app.logger.print_message(f"{exc}: Укажите все параметры "
                                          f"виртуальных испытаний "
                                          f"в виде чисел.")

    def create_param_dict(self):
        self.collect_input_data()
        self.param_dict = {"strut-parameters": self.strut_dict,
                           "tyre-parameters": self.tyre_dict,
                           "test-parameters": self.test_dict}
    
    def show_parameters(self):
        if not self.strut_dict or not self.tyre_dict:
            self.app.push_screen(PopUpScreenMessage("Параметры модели не загружены!"))
            return
        
        self.create_param_dict()
        self.app.push_screen(ParametersView(self.param_dict))


class PostProc(ScrollableContainer):
    DT_COLCOUNT = 11
    DT_CELL_WIDTH = 11
    def __init__(self):
        super().__init__()
        self.first_strip = InputWithAction(input_label="Открыть файл",
                                           button_label="Открыть")
        self.second_strip = InputWithAction(input_label="Сохранить файл",
                                            button_label="Сохранить")
        self.dt = self.create_data_table()
        self.data = [[] for _ in range(self.DT_COLCOUNT)]
        self.data_loaded = False
        self.request_clear = True
    
    def create_data_table(self) -> DataTable:
        self._dt = DataTable()
        self._dt.zebra_stripes = True
        self._dt.cursor_type   = "row"
        self._dt.fixed_columns = 1
        self._dt.add_column("  t, c  ", width=self.DT_CELL_WIDTH-3)
        self._dt.add_column("   u, мм   ", width=self.DT_CELL_WIDTH)
        self._dt.add_column("   v, мм   ", width=self.DT_CELL_WIDTH)
        self._dt.add_column(" ω₁, рад/с ", width=self.DT_CELL_WIDTH)
        self._dt.add_column("     s     ", width=self.DT_CELL_WIDTH)
        self._dt.add_column("   y₁, мм  ", width=self.DT_CELL_WIDTH)
        self._dt.add_column("   y, мм   ", width=self.DT_CELL_WIDTH)
        self._dt.add_column("   Fx, т   ", width=self.DT_CELL_WIDTH)
        self._dt.add_column("   Fy, т   ", width=self.DT_CELL_WIDTH)
        self._dt.add_column("  Mz, т·м  ", width=self.DT_CELL_WIDTH)
        self._dt.add_column("  p₁, атм  ", width=self.DT_CELL_WIDTH)
        return self._dt
        
    def compose(self) -> ComposeResult:
        """Composes and yields containers for file results and a datatable.

        This method creates a container with two strips, "first_strip" and 
        "second_strip", and yields it. The container has a border title and 
        aligned styles. It then yields a container with a datatable, "dt".

        Returns:
            ComposeResult: The composed containers.
        """
        cnt = Container(self.first_strip,
                        self.second_strip,
                        id="container-file-results")
        cnt.border_title = "Работа с файлами результатов"
        cnt.styles.border_title_align = "center"
        yield cnt
        yield Container(self.dt, id="container-datatable")
    
    @staticmethod        
    def change_button_color(button, success) -> None:
        if success:
            button.remove_class("failed")
            button.add_class("succeeded")
        else:
            button.remove_class("succeeded")
            button.add_class("failed")
            
    def request_load_data(self) -> None:
        self.request_clear = True
        def check_clear(clear: bool) -> None:
            if clear:
                self.dt.clear()
                self.app.logger.print_message("Данные удалены!")
                self.load_data()

        if self.data_loaded:
            self.app.push_screen(PopUpScreenChoice("Результаты будут удалены!"), 
                                 check_clear)
        else:
            self.load_data()
        
    def load_data(self) -> None:    
        fname = self.first_strip.input.value
        
        if not fname:
            self.change_button_color(self.first_strip.button, False)
            return
        self.app.logger.print_message(f"Открытие файла "
                                      f"{os.path.basename(fname)}...")
        try:
            with open(fname) as file:
                data = np.loadtxt(file)
        except Exception as exc:
            self.app.logger.print_message(f"{exc}: Ошибка при открытии файла "
                                          f"{os.path.basename(fname)}.")
            self.first_strip.input.value = ""
            self.change_button_color(self.first_strip.button, False)
            return
        
        size_kB = data[1:].size * data[1:].itemsize / 1024
        
        self.app.logger.print_message(f"Файл {os.path.basename(fname)} " 
                                      f"успешно загружен. "
                                  f"Размер: {data[1:].shape[0]}×"
                                  f"{data[1:].shape[1]} "
                                  f"({(size_kB):3.2f} кБ).")
        
        self.change_button_color(self.first_strip.button, True)
            
        self.data_loaded = True
        self.fill_data_table(data)
        
    def update_cells(self, data):
        for i, data_row in enumerate(data):
            label = Text(str(i+1), style="#03AC13 italic", justify="center")
            cells = [Text(f"{value:.2E}" if i == 0 else f"{value: .3E}", 
                        style="#38B6FF", 
                        justify="center") for i, value in enumerate(data_row)]
            self.dt.add_row(*cells, label=label)

    def add_data_info(self, data):
        mins = [data_col.min() for data_col in data.T]
        maxs = [data_col.max() for data_col in data.T]

        self.dt.add_row(*[Text(f"{value: .3E}", 
                            style="#03AC13 italic", 
                            justify="center") for value in mins], 
                        label=Text("Min",
                                style="#03AC13 italic",
                                justify="center"))
        self.dt.add_row(*[Text(f"{value: .3E}", 
                            style="#FF0090 italic", 
                            justify="center") for value in maxs], 
                        label=Text("Max", style="#FF0090 italic",
                                justify="center"))
    
    def fill_data_table(self, data) -> None:
        self.update_cells(data)
        self.add_data_info(data)
        self.app.logger.print_message("Таблица заполнена.")
        
    def request_save_data(self) -> None:
        fname = self.second_strip.input.value
        
        if not fname:
            self.change_button_color(self.second_strip.button, False)
            return
        try:
            np.savetxt(fname, self.data)
        except Exception as exc:
            self.app.logger.print_message(f"{exc}: "
                                          f"Ошибка при сохранении файла.")
            self.change_button_color(self.second_strip.button, False)
            return
        self.app.logger.print_message(f"Файл {fname} успешно сохранен.")
        self.change_button_color(self.second_strip.button, True)
    

class Logger(RichLog):
    id = "log"
    BORDER_TITLE = "Распечатка действий"
    def print_message(self, message):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        self.write(f"{dt_string} {message}")
