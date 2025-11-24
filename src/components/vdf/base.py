from abc import ABC, abstractmethod

class BaseVDF(ABC):
    @abstractmethod
    def calculate(self, flow, capacity):
        """Todas las VDF deben recibir flow y capacity"""
        pass