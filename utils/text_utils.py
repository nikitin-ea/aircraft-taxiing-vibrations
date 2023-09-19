import os
import glob
import natsort
from datetime import datetime

def print_msg(text: str):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(dt_string + " " + text)

def scan_directory(path: str, file_type: str) -> list:
    os.chdir(path)
    filenames = natsort.natsorted(glob.glob(f"*.{file_type}"))
    return filenames
