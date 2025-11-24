import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict

class TrafficDataset(Dataset):
    """
    Dataset para entrenamiento con datos parciales.
    Maneja flujos observados, demandas OD y sus respectivas máscaras.
    """

    def __init__(self,
                 true_flows: np.ndarray,
                 true_od: np.ndarray,
                 flow_mask: np.ndarray,
                 od_mask: np.ndarray):
        """
        Args:
            true_flows: Flujos verdaderos [num_samples, num_links]
            true_od: Demandas OD verdaderas [num_samples, num_od_pairs]
            flow_mask: Máscaras de flujos [num_samples, num_links] (1=observado, 0=desconocido)
            od_mask: Máscaras de OD [num_samples, num_od_pairs] (1=conocido, 0=desconocido)
        """
        # Convertimos a tensores de PyTorch asegurando tipo Float
        self.true_flows = torch.FloatTensor(true_flows)
        self.true_od = torch.FloatTensor(true_od)
        self.flow_mask = torch.FloatTensor(flow_mask)
        self.od_mask = torch.FloatTensor(od_mask)

    def __len__(self) -> int:
        return len(self.true_flows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'true_flows': self.true_flows[idx],
            'true_od': self.true_od[idx],
            'flow_mask': self.flow_mask[idx],
            'od_mask': self.od_mask[idx]
        }