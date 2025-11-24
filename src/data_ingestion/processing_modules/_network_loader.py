"""
TNTP network loader moved to processing_modules to separate concerns.
"""
# ... reuse content from original _network_loader.py but adjust PROJECT_ROOT resolution
import pandas as pd
from pathlib import Path
import warnings
from typing import Tuple, Dict, List, Optional, Union
import io

warnings.filterwarnings('ignore')


class TNTPNetworkLoader:
    def __init__(self, path: Optional[Union[Path, str]] = None):
        if path:
            self.path = Path(path)
        else:
            project_root = Path(__file__).resolve().parents[2]
            self.path = project_root / 'data' / 'raw' / 'Barcelona_net.tntp'
        self.path = self.path.resolve(strict=False)
        self._exists = self.path.exists()

    def _read_lines(self) -> List[str]:
        if not self._exists:
            raise FileNotFoundError(f"Archivo no encontrado: {self.path}")
        with open(self.path, 'r', encoding='utf-8', errors='replace') as f:
            return f.readlines()

    def _parse_metadata(self, lines: List[str]) -> Dict[str, Union[int, List[str]]]:
        metadata = {}
        tags = {
            '<NUMBER OF ZONES>': 'zones',
            '<NUMBER OF NODES>': 'nodes',
            '<FIRST THRU NODE>': 'first_thru_node',
            '<NUMBER OF LINKS>': 'links'
        }
        for line in lines:
            line = line.strip()
            for tag, key in tags.items():
                if line.startswith(tag):
                    try:
                        metadata[key] = int(line.split()[-1])
                    except Exception:
                        pass
            if line.startswith('<ORIGINAL HEADER>') and '~' in line:
                header = line.split('~', 1)[1].strip()
                metadata['columns'] = [col.strip() for col in header.split('\t') if col.strip()]
            if line == '<END OF METADATA>':
                break
        return metadata

    def _find_data_start(self, lines: List[str]) -> int:
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('~') and 'init_node' in stripped.lower() and 'term_node' in stripped.lower():
                return i + 1
            if 'init_node' in stripped.lower() and 'term_node' in stripped.lower():
                return i + 1
        for i, line in enumerate(lines):
            if line.strip().endswith(';'):
                return i
        raise ValueError('No se encontró el inicio de los datos en el archivo TNTP')

    def _prepare_data_lines(self, lines: List[str], start_idx: int) -> List[str]:
        data_lines = []
        for line in lines[start_idx:]:
            line = line.strip()
            if not line or line.startswith('~'):
                continue
            if line.endswith(';'):
                data_lines.append(line[:-1].strip())
            elif any(c.isdigit() for c in line) and ('\t' in line or '  ' in line):
                data_lines.append(line)
        return data_lines

    def _parse_dataframe(self, data_lines: List[str]) -> pd.DataFrame:
        if not data_lines:
            raise ValueError('No se encontraron líneas de datos en el archivo')
        data_string = '\n'.join(data_lines)
        raw = pd.read_csv(io.StringIO(data_string), sep=r'\s+', header=None, engine='python')
        ncols = raw.shape[1]
        base_cols = ['init_node', 'term_node', 'capacity', 'length',
                     'free_flow_time', 'b', 'power', 'speed', 'toll', 'link_type']
        cols = None
        if ncols == len(base_cols):
            cols = base_cols
        elif ncols == len(base_cols) + 1:
            col3 = raw.iloc[:, 3]
            try:
                frac_small_int = (col3.dropna().apply(float).round(0) == col3.dropna().astype(float)).sum() / max(1, len(col3.dropna()))
            except Exception:
                frac_small_int = 0.0
            median_col3 = float(col3.dropna().median()) if len(col3.dropna()) > 0 else float('nan')
            if (not pd.isna(median_col3) and median_col3 <= 20 and frac_small_int > 0.6):
                cols = ['init_node', 'term_node', 'capacity', 'lanes'] + base_cols[3:]
            else:
                cols = base_cols[:9] + ['VDF'] + base_cols[9:]
        elif ncols == len(base_cols) + 2:
            cols = ['init_node', 'term_node', 'capacity', 'lanes'] + base_cols[3:9] + ['VDF'] + [base_cols[9]]
        else:
            cols = []
            for i in range(ncols):
                if i < len(base_cols):
                    cols.append(base_cols[i])
                else:
                    cols.append(f'col_extra_{i}')
        raw.columns = cols
        for col in raw.columns:
            if col in ['lanes']:
                raw[col] = pd.to_numeric(raw[col], errors='coerce')
                raw[col] = raw[col].fillna(1).astype(int)
            elif col == 'VDF':
                raw[col] = pd.to_numeric(raw[col], errors='coerce')
            else:
                raw[col] = pd.to_numeric(raw[col], errors='coerce')
        return raw

    def load(self) -> Tuple[pd.DataFrame, Dict[str, Union[int, List[str]]]]:
        lines = self._read_lines()
        metadata = self._parse_metadata(lines)
        start_idx = self._find_data_start(lines)
        data_lines = self._prepare_data_lines(lines, start_idx)
        df = self._parse_dataframe(data_lines)
        return df, metadata


def load_network_df(path: Optional[Union[Path, str]] = None) -> pd.DataFrame:
    loader = TNTPNetworkLoader(path)
    df, _ = loader.load()
    return df

