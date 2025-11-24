"""
DataSaver - responsable de guardar salidas (grafo, imagen, od matrix, unified df)

Provee funciones para persistir los objetos procesados por DataManager.
"""
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

class DataSaver:
    """
    Handles the persistence (saving) of processed data objects, such as
    network graphs, unified DataFrames, and OD matrices, to the file system.
    """
    def __init__(self, processed_dir: Union[str, Path]):
        """
        Initializes the DataSaver, setting up the base directory for output files.

        :param processed_dir: The root directory where all processed files will be saved.
        """
        self.processed_dir = Path(processed_dir)
        # Create the directory if it doesn't exist (and any necessary parents)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def save_graph_pickle(self, graph: nx.Graph, file_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Saves a NetworkX graph object using pickle serialization.

        It attempts to use NetworkX's specialized gpickle first, falling back
        to Python's standard pickle module if necessary.

        :param graph: The networkx.Graph object to save.
        :param file_path: Optional, the specific path to save the file to.
                          Defaults to 'graph.pkl' in the processed directory.
        :return: The Path object of the saved file.
        """
        if file_path is None:
            file_path = self.processed_dir / 'graph.pkl'
        else:
            file_path = Path(file_path)

        try:
            # Try specialized NetworkX pickle
            from networkx.readwrite import gpickle
            gpickle.write_gpickle(graph, file_path)
            return file_path
        except Exception:
            # Fallback to standard Python pickle
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump(graph, f)
            return file_path

    def save_graph_image(self, graph: nx.Graph, file_path: Optional[Union[str, Path]] = None, fmt: str = 'png') -> Path:
        """
        Renders the graph and saves it as an image file (e.g., PNG, SVG, PDF).

        :param graph: The networkx.Graph object to visualize.
        :param file_path: Optional, the specific path for the image file.
                          Defaults to 'graph.{fmt}' in the processed directory.
        :param fmt: The desired image format (e.g., 'png', 'svg', 'pdf').
        :return: The Path object of the saved image file.
        """
        if file_path is None:
            file_path = self.processed_dir / f'graph.{fmt}'
        else:
            file_path = Path(file_path)

        # Calculate node positions using the spring layout algorithm
        pos = nx.spring_layout(graph)
        # Draw the graph
        nx.draw(graph, pos, with_labels=True, node_size=50, font_size=8)
        # Save the drawn figure
        plt.savefig(file_path, format=fmt)
        # Close the plot to free memory
        plt.close()
        return file_path

    def save_unified_df(self, df: pd.DataFrame, file_path: Optional[Union[str, Path]] = None, fmt: str = 'parquet') -> Path:
        """
        Saves the processed Pandas DataFrame to disk in a specified format.

        Supported formats are Parquet (efficient binary), CSV (text), or Pickle (fallback).

        :param df: The pandas.DataFrame to save (e.g., unified network data).
        :param file_path: Optional, the specific path for the file.
                          Defaults to 'unified.{fmt}' in the processed directory.
        :param fmt: The storage format ('parquet', 'csv'). Defaults to 'parquet'.
        :return: The Path object of the saved file.
        """
        if file_path is None:
            file_path = self.processed_dir / f'unified.{fmt}'
        else:
            file_path = Path(file_path)

        if fmt == 'parquet':
            df.to_parquet(file_path)
        elif fmt == 'csv':
            df.to_csv(file_path, index=False)
        else:
            # Default to pickle for unsupported formats
            df.to_pickle(file_path)
        return file_path

    def save_od_matrix(self, od_matrix, file_path: Optional[Union[str, Path]] = None, fmt: str = 'sparse') -> Path:
        """
        Saves the Origin-Destination matrix to disk.

        Handles sparse (SciPy NPZ) and dense (NumPy) formats.

        :param od_matrix: The OD matrix object (can be a SciPy sparse matrix or NumPy array).
        :param file_path: Optional, the specific path for the file.
                          Defaults to 'od_matrix.{fmt}' in the processed directory.
        :param fmt: The storage format ('sparse', 'dense'). Defaults to 'sparse'.
        :return: The Path object of the saved file.
        """
        if file_path is None:
            file_path = self.processed_dir / f'od_matrix.{fmt}'
        else:
            file_path = Path(file_path)

        from scipy import sparse
        import numpy as np

        # Auto-detect format if matrix type doesn't match requested format
        is_sparse = sparse.issparse(od_matrix)

        if fmt == 'sparse':
            if is_sparse:
                # Already sparse, save directly
                sparse.save_npz(file_path, od_matrix)
            else:
                # Dense array, convert to sparse and save
                od_matrix_sparse = sparse.csr_matrix(od_matrix)
                sparse.save_npz(file_path, od_matrix_sparse)
        elif fmt == 'dense':
            # Save as a standard NumPy array
            # Convert sparse matrix to dense array if necessary before saving
            data_to_save = od_matrix.toarray() if is_sparse else od_matrix
            np.save(file_path, data_to_save)
        else:
            # Default to pickle for other formats
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump(od_matrix, f)
        return file_path
