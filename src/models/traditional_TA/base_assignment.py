"""
Clase base para algoritmos de Traffic Assignment.

Define la interfaz común para todos los algoritmos de asignación de tráfico.
"""
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
from typing import Dict, Tuple


class BaseTrafficAssignment:
    """
    Clase base para algoritmos de Traffic Assignment.

    Attributes:
        graph: Grafo de la red de transporte
        od_matrix: Matriz origen-destino (sparse)
        cost_attr: Atributo del grafo a usar como costo
        solution_attr: Atributo donde guardar la solución
        iteration_history: Historial de iteraciones
        convergence_gap: Gap de convergencia final
        iterations: Número de iteraciones ejecutadas
    """

    def __init__(self,
                 graph: nx.DiGraph,
                 od_matrix: sparse.spmatrix,
                 cost_attr: str = 'length',
                 solution_attr: str = 'solution_nocongestion'):
        """
        Inicializa el modelo de Traffic Assignment.

        Args:
            graph: Grafo NetworkX con la red de transporte
            od_matrix: Matriz OD sparse
            cost_attr: Atributo a usar como costo
            solution_attr: Atributo donde guardar la solución
        """
        self.graph = graph
        self.od_matrix = od_matrix
        self.cost_attr = cost_attr
        self.solution_attr = solution_attr

        # Inicializar flujos en cero
        for u, v in self.graph.edges():
            self.graph[u][v][solution_attr] = 0.0

        # Historial
        self.iteration_history = []
        self.convergence_gap = None
        self.iterations = 0

    def get_edge_cost(self, u: int, v: int) -> float:
        """Obtiene el costo de un enlace."""
        return self.graph[u][v].get(self.cost_attr, 0.0)

    def get_edge_flow(self, u: int, v: int) -> float:
        """Obtiene el flujo actual de un enlace."""
        return self.graph[u][v].get(self.solution_attr, 0.0)

    def compute_total_cost(self) -> float:
        """Calcula el costo total del sistema."""
        total_cost = 0.0
        for u, v in self.graph.edges():
            cost = self.get_edge_cost(u, v)
            flow = self.get_edge_flow(u, v)
            total_cost += cost * flow
        return total_cost

    def compare_with_observed(self, observed_attr: str = 'volume'):
        """
        Compara la solución con flujos observados.

        Args:
            observed_attr: Atributo con flujos observados

        Returns:
            DataFrame con comparación
        """
        data = []
        for u, v, edge_data in self.graph.edges(data=True):
            observed = edge_data.get(observed_attr, 0.0)
            predicted = edge_data.get(self.solution_attr, 0.0)

            data.append({
                'init_node': u,
                'term_node': v,
                'observed': observed,
                'predicted': predicted,
                'error': predicted - observed,
                'abs_error': abs(predicted - observed),
                'pct_error': abs(predicted - observed) / observed * 100 if observed > 0 else 0
            })

        df = pd.DataFrame(data)

        # Calcular métricas
        mae = df['abs_error'].mean()
        rmse = np.sqrt((df['error'] ** 2).mean())
        mape = df['pct_error'].mean()

        df.attrs['MAE'] = mae
        df.attrs['RMSE'] = rmse
        df.attrs['MAPE'] = mape
        df.attrs['total_observed'] = df['observed'].sum()
        df.attrs['total_predicted'] = df['predicted'].sum()

        return df

    def save_comparison(self, output_path: Path, observed_attr: str = 'volume'):
        """
        Guarda la comparación en un archivo CSV.

        Args:
            output_path: Ruta del archivo de salida
            observed_attr: Atributo con flujos observados

        Returns:
            Path del archivo guardado
        """
        df = self.compare_with_observed(observed_attr)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path

