from .base import BaseVDF

class BPRFunction(BaseVDF):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def calculate(self, flow, capacity):
        # Fórmula BPR estándar
        return 1 + self.alpha * ((flow / capacity) ** self.beta)