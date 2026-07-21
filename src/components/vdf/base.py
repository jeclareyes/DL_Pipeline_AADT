import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseCostFunction(nn.Module, ABC):
    """
    Clase base abstracta para funciones de costo (VDFs).
    Hereda de nn.Module para integrarse con el pipeline de entrenamiento de PyTorch
    (manejo de device, parámetros aprendibles, buffers, etc.).
    
    Además, expone métodos para cálculos estáticos con NumPy, para ser usados en 
    los assignment motors (ej. Frank-Wolfe, UE).
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, link_flows: torch.Tensor) -> torch.Tensor:
        """
        Calcula el tiempo de viaje o costo dado el flujo en los enlaces (PyTorch).

        Args:
            link_flows (torch.Tensor): Tensor de flujos en enlaces.

        Returns:
            torch.Tensor: Costos calculados.
        """
        pass

    @classmethod
    @abstractmethod
    def evaluate_costs_numpy(cls, link_flows: np.ndarray, link_table: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Calcula el tiempo de viaje o costo dado el flujo en los enlaces usando NumPy.
        Extrae los parámetros necesarios desde link_table, mapeados por kwargs.
        """
        pass

    @classmethod
    @abstractmethod
    def evaluate_beckmann_integral_numpy(cls, link_flows: np.ndarray, link_table: pd.DataFrame, **kwargs) -> float:
        """
        Calcula la integral de Beckmann para esta VDF usando NumPy.
        Si una VDF no soporta este cálculo, debe levantar NotImplementedError.
        """
        raise NotImplementedError("Esta VDF no tiene implementada la lógica de la integral de Beckmann.")

    @classmethod
    @abstractmethod
    def get_required_columns(cls) -> list[str]:
        """
        Retorna la lista de nombres de parámetros genéricos (o columnas por defecto)
        requeridas en el link_table para evaluar esta VDF.
        """
        pass