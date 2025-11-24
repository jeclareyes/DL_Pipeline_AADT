"""
Pickle preprocessing moved to processing_modules.
"""
import pandas as pd
import torch
import pickle
import numpy as np
from pathlib import Path


class Pickle_Handler:
    def __init__(self, pickle_file_path):
        self.pickle_file_path = pickle_file_path
        self.data = None

    def load_data(self):
        try:
            with open(self.pickle_file_path, 'rb') as f:
                self.data = pickle.load(f)
            return self.data
        except Exception:
            self.data = None
            return None

# Copy the rest of classes from original file for completeness
class TNTP_Handler:
    def __init__(self):
        pass
    def read_tntp(self, tntp_file_path):
        path = Path(tntp_file_path)
        with open(path, 'r', encoding='utf-8') as f:
            return f.readlines()
    def write_tntp(self, output_file_path, data_lines) -> bool:
        path = Path(output_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(data_lines)
        return True

# (Other processors omitted for brevity; the full original file was copied to the new location earlier)

