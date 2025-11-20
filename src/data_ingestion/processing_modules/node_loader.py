"""Node loader moved into processing_modules."""
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, Union, List
import warnings

warnings.filterwarnings('ignore')


class TNTPNodeLoader:
    """
    Loads node data from a TNTP format file (typically 'nodes.tntp').

    This loader is designed to parse lines containing Node ID, X-coordinate,
    Y-coordinate, and an optional Type field.
    """

    def __init__(self, path: Optional[Union[Path, str]] = None):
        """
        Initializes the loader with the file path.

        :param path: The explicit path to the TNTP node file.
                     If None, a default path relative to the project structure is assumed.
        """
        if path:
            self.path = Path(path)
        else:
            # Assumes a specific default path structure if none is provided
            project_root = Path(__file__).resolve().parents[2]
            self.path = project_root / 'data' / 'raw' / 'nodes.tntp'

        self.path = self.path.resolve(strict=False)
        self._exists = self.path.exists()

    def _read_lines(self) -> List[str]:
        """Reads all lines from the file, raising FileNotFoundError if the file is missing."""
        if not self._exists:
            raise FileNotFoundError(f"File not found: {self.path}")
        with open(self.path, 'r', encoding='utf-8', errors='replace') as f:
            return f.readlines()

    def _find_data_start(self, lines: List[str]) -> int:
        """
        Identifies the index (line number) where the actual node data begins.

        It looks for a header line that typically contains 'Node' and a coordinate name ('X' or 'x').

        :param lines: List of all lines read from the file.
        :return: The index of the line immediately following the header.
        """
        for i, line in enumerate(lines):
            stripped = line.strip()
            if 'Node' in stripped and ('X' in stripped or 'x' in stripped):
                return i + 1
        return 0

    def _prepare_data_lines(self, lines: List[str], start_idx: int) -> List[str]:
        """
        Filters and cleans the raw lines to isolate valid data entries.

        It skips comments (#, //), empty lines, and removes the trailing semicolon (;).

        :param lines: List of all lines read from the file.
        :param start_idx: The line index where data is expected to start.
        :return: A list of clean data strings ready for parsing.
        """
        data_lines = []
        for line in lines[start_idx:]:
            stripped = line.strip()
            # Skip empty lines or comment lines
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                continue
            # Remove the trailing semicolon typical in TNTP format
            if stripped.endswith(';'):
                stripped = stripped[:-1].strip()
            data_lines.append(stripped)
        return data_lines

    def _parse_data(self, data_lines: List[str]) -> pd.DataFrame:
        """
        Parses clean data lines into a Pandas DataFrame.

        Handles both 4-column format (Node, X, Y, Type) and 5-column format (Index, Node ID, X, Y, Type).

        :param data_lines: List of clean data strings.
        :return: A DataFrame containing the columns 'node', 'x', 'y', and 'type'.
        """
        nodes = []
        for line in data_lines:
            parts = line.split()
            # Handle both 4-column and 5-column formats
            if len(parts) >= 5:
                # 5-column format: Index, Node ID, X, Y, Type
                try:
                    node = int(parts[1])  # Use second column as node ID
                    x = float(parts[2])
                    y = float(parts[3])
                    node_type = parts[4].strip()

                    nodes.append({
                        'node': node,
                        'x': x,
                        'y': y,
                        'type': node_type
                    })
                except (ValueError, IndexError):
                    continue
            elif len(parts) >= 4:
                # 4-column format: Node, X, Y, Type
                try:
                    node = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    node_type = parts[3].strip()

                    nodes.append({
                        'node': node,
                        'x': x,
                        'y': y,
                        'type': node_type
                    })
                except (ValueError, IndexError):
                    continue
        return pd.DataFrame(nodes)

    def load(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Main method to load node data and calculate basic metadata.

        :return: A tuple containing:
                 1. pd.DataFrame: The node coordinates and type data.
                 2. Dict: Metadata including node count and coordinate ranges.
        """
        if not self._exists:
            raise FileNotFoundError(f"File not found: {self.path}")

        lines = self._read_lines()
        start_idx = self._find_data_start(lines)
        data_lines = self._prepare_data_lines(lines, start_idx)

        nodes_df = self._parse_data(data_lines)

        # Calculate metadata
        metadata = {
            'num_nodes': len(nodes_df),
            'x_range': (nodes_df['x'].min(), nodes_df['x'].max()) if not nodes_df.empty else (None, None),
            'y_range': (nodes_df['y'].min(), nodes_df['y'].max()) if not nodes_df.empty else (None, None),
            'file': str(self.path)
        }
        return nodes_df, metadata


# --------------------------------------------------------------------------

def load_node_features(path: Union[Path, str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to load node data using the TNTPNodeLoader class.

    :param path: The path to the TNTP node file.
    :return: A tuple of (DataFrame of nodes, Metadata dictionary).
    """
    loader = TNTPNodeLoader(path)
    return loader.load()

