import torch.nn as nn
from abc import ABC, abstractmethod


class BaseCostFunction(nn.Module, ABC):
    """
    Clase base abstracta para funciones de costo (VDFs).
    Hereda de nn.Module para integrarse con el pipeline de entrenamiento de PyTorch
    (manejo de device, parámetros aprendibles, buffers, etc.).
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, link_flows):
        """
        Calcula el tiempo de viaje o costo dado el flujo en los enlaces.

        Args:
            link_flows (torch.Tensor): Tensor de flujos en enlaces.

        Returns:
            torch.Tensor: Costos calculados.
        """
        pass