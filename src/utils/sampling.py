"""
Estrategias de muestreo para datos parciales de OD y flujos.

Este módulo implementa diferentes técnicas de muestreo para simular
disponibilidad parcial de datos, generando máscaras binarias (0/1) que
indican qué datos están disponibles/conocidos.

Estrategias implementadas:
- Random: muestreo aleatorio uniforme
- Stratified: muestreo estratificado por magnitud
- Spatial: muestreo basado en proximidad espacial (futuro)
- Temporal: muestreo basado en patrones temporales (futuro)
"""
import torch
import numpy as np
from typing import Optional


class DataSampler:
    """
    Clase base para estrategias de muestreo de datos parciales.

    Genera máscaras binarias (0/1) para indicar datos conocidos/desconocidos:
    - 1: dato conocido/observado
    - 0: dato desconocido/missing
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: Semilla para reproducibilidad
        """
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def sample_od_mask(self, od_demand: torch.Tensor, od_rate: float) -> torch.Tensor:
        """
        Genera máscara de muestreo para matriz OD.

        Args:
            od_demand: Tensor de demandas OD [batch_size, num_od_pairs] o [num_od_pairs]
            od_rate: Fracción de pares OD a observar (0.0 a 1.0)

        Returns:
            Máscara binaria del mismo shape que od_demand (1=conocido, 0=desconocido)
        """
        raise NotImplementedError

    def sample_flow_mask(self, flows: torch.Tensor, flow_rate: float) -> torch.Tensor:
        """
        Genera máscara de muestreo para flujos de enlaces.

        Args:
            flows: Tensor de flujos [batch_size, num_links] o [num_links]
            flow_rate: Fracción de enlaces a observar (0.0 a 1.0)

        Returns:
            Máscara binaria del mismo shape que flows (1=conocido, 0=desconocido)
        """
        raise NotImplementedError


class RandomSampler(DataSampler):
    """
    Muestreo aleatorio uniforme.

    Cada elemento tiene probabilidad `rate` de ser observado,
    independientemente de su magnitud o posición.
    """

    def sample_od_mask(self, od_demand: torch.Tensor, od_rate: float) -> torch.Tensor:
        """Muestreo aleatorio uniforme para OD."""
        mask = torch.rand_like(od_demand) < od_rate
        return mask.float()

    def sample_flow_mask(self, flows: torch.Tensor, flow_rate: float) -> torch.Tensor:
        """Muestreo aleatorio uniforme para flujos."""
        mask = torch.rand_like(flows) < flow_rate
        return mask.float()


class StratifiedSampler(DataSampler):
    """
    Muestreo estratificado por magnitud.

    Divide los datos en estratos (quintiles) según su magnitud
    y muestrea proporcionalmente de cada estrato.

    Ventaja: Asegura representación de valores altos, medios y bajos.
    """

    def __init__(self, num_strata: int = 5, seed: Optional[int] = None):
        """
        Args:
            num_strata: Número de estratos (típicamente 5 para quintiles)
            seed: Semilla para reproducibilidad
        """
        super().__init__(seed)
        self.num_strata = num_strata

    def sample_od_mask(self, od_demand: torch.Tensor, od_rate: float) -> torch.Tensor:
        """Muestreo estratificado para OD."""
        return self._stratified_sample(od_demand, od_rate)

    def sample_flow_mask(self, flows: torch.Tensor, flow_rate: float) -> torch.Tensor:
        """Muestreo estratificado para flujos."""
        return self._stratified_sample(flows, flow_rate)

    def _stratified_sample(self, data: torch.Tensor, rate: float) -> torch.Tensor:
        """
        Implementación genérica de muestreo estratificado.

        Args:
            data: Tensor de datos
            rate: Tasa de muestreo global

        Returns:
            Máscara binaria
        """
        original_shape = data.shape
        is_batched = data.dim() == 2

        if not is_batched:
            data = data.unsqueeze(0)

        batch_size, num_elements = data.shape
        mask = torch.zeros_like(data)

        for b in range(batch_size):
            values = data[b]

            # Dividir en estratos basados en percentiles
            non_zero_mask = values > 0
            if non_zero_mask.sum() == 0:
                # Si todos son cero, muestreo aleatorio
                batch_mask = torch.rand(num_elements, device=data.device) < rate
                mask[b] = batch_mask.float()
                continue

            non_zero_values = values[non_zero_mask]

            # Calcular percentiles para estratificación
            percentiles = torch.linspace(0, 100, self.num_strata + 1, device=data.device)
            thresholds = torch.quantile(non_zero_values, percentiles / 100.0)

            # Asignar cada elemento a un estrato
            strata_assignment = torch.zeros(num_elements, dtype=torch.long, device=data.device)
            for i in range(self.num_strata):
                if i == self.num_strata - 1:
                    # Último estrato incluye el máximo
                    in_stratum = (values >= thresholds[i]) & (values <= thresholds[i + 1])
                else:
                    in_stratum = (values >= thresholds[i]) & (values < thresholds[i + 1])
                strata_assignment[in_stratum] = i

            # Muestrear proporcionalmente de cada estrato
            batch_mask = torch.zeros(num_elements, dtype=torch.bool, device=data.device)
            for stratum in range(self.num_strata):
                in_stratum = strata_assignment == stratum
                stratum_size = in_stratum.sum().item()

                if stratum_size > 0:
                    # Número de elementos a muestrear de este estrato
                    n_sample = max(1, int(stratum_size * rate))

                    # Índices del estrato
                    stratum_indices = torch.where(in_stratum)[0]

                    # Muestreo aleatorio dentro del estrato
                    perm = torch.randperm(stratum_size, device=data.device)
                    sampled_indices = stratum_indices[perm[:n_sample]]

                    batch_mask[sampled_indices] = True

            mask[b] = batch_mask.float()

        if not is_batched:
            mask = mask.squeeze(0)

        return mask


class TopKSampler(DataSampler):
    """
    Muestreo de los K elementos con mayor magnitud.

    Útil para simular sensores colocados en los enlaces/OD más importantes.
    """

    def sample_od_mask(self, od_demand: torch.Tensor, od_rate: float) -> torch.Tensor:
        """Muestreo de top-K OD pairs por demanda."""
        return self._topk_sample(od_demand, od_rate)

    def sample_flow_mask(self, flows: torch.Tensor, flow_rate: float) -> torch.Tensor:
        """Muestreo de top-K enlaces por flujo."""
        return self._topk_sample(flows, flow_rate)

    def _topk_sample(self, data: torch.Tensor, rate: float) -> torch.Tensor:
        """
        Implementación genérica de muestreo top-K.

        Args:
            data: Tensor de datos
            rate: Tasa de muestreo (fracción de top elementos)

        Returns:
            Máscara binaria
        """
        original_shape = data.shape
        is_batched = data.dim() == 2

        if not is_batched:
            data = data.unsqueeze(0)

        batch_size, num_elements = data.shape
        k = max(1, int(num_elements * rate))

        # Top-K por batch
        _, topk_indices = torch.topk(data, k, dim=1)

        # Crear máscara
        mask = torch.zeros_like(data)
        mask.scatter_(1, topk_indices, 1.0)

        if not is_batched:
            mask = mask.squeeze(0)

        return mask


class HybridSampler(DataSampler):
    """
    Muestreo híbrido: combina top-K con aleatorio.

    Asegura que se observen los elementos más importantes,
    mientras agrega diversidad con muestreo aleatorio del resto.
    """

    def __init__(self, topk_ratio: float = 0.5, seed: Optional[int] = None):
        """
        Args:
            topk_ratio: Fracción de la muestra dedicada a top-K (resto es aleatorio)
            seed: Semilla para reproducibilidad
        """
        super().__init__(seed)
        self.topk_ratio = topk_ratio

    def sample_od_mask(self, od_demand: torch.Tensor, od_rate: float) -> torch.Tensor:
        """Muestreo híbrido para OD."""
        return self._hybrid_sample(od_demand, od_rate)

    def sample_flow_mask(self, flows: torch.Tensor, flow_rate: float) -> torch.Tensor:
        """Muestreo híbrido para flujos."""
        return self._hybrid_sample(flows, flow_rate)

    def _hybrid_sample(self, data: torch.Tensor, rate: float) -> torch.Tensor:
        """
        Implementación genérica de muestreo híbrido.

        Args:
            data: Tensor de datos
            rate: Tasa de muestreo global

        Returns:
            Máscara binaria
        """
        original_shape = data.shape
        is_batched = data.dim() == 2

        if not is_batched:
            data = data.unsqueeze(0)

        batch_size, num_elements = data.shape
        total_sample = max(1, int(num_elements * rate))

        # Dividir muestra entre top-K y aleatorio
        k_topk = max(1, int(total_sample * self.topk_ratio))
        k_random = total_sample - k_topk

        mask = torch.zeros_like(data)

        for b in range(batch_size):
            # Top-K
            _, topk_indices = torch.topk(data[b], k_topk)
            mask[b, topk_indices] = 1.0

            # Aleatorio del resto (excluyendo top-K)
            if k_random > 0:
                remaining_indices = torch.ones(num_elements, dtype=torch.bool, device=data.device)
                remaining_indices[topk_indices] = False
                remaining_indices = torch.where(remaining_indices)[0]

                if len(remaining_indices) > 0:
                    perm = torch.randperm(len(remaining_indices), device=data.device)
                    random_sample = remaining_indices[perm[:k_random]]
                    mask[b, random_sample] = 1.0

        if not is_batched:
            mask = mask.squeeze(0)

        return mask


def get_sampler(strategy: str, **kwargs) -> DataSampler:
    """
    Factory function para crear samplers.

    Args:
        strategy: Nombre de la estrategia ('random', 'stratified', 'topk', 'hybrid')
        **kwargs: Argumentos adicionales para el sampler

    Returns:
        Instancia de DataSampler

    Example:
        >>> sampler = get_sampler('stratified', num_strata=5, seed=42)
        >>> od_mask = sampler.sample_od_mask(od_demand, od_rate=0.2)
    """
    samplers = {
        'random': RandomSampler,
        'stratified': StratifiedSampler,
        'topk': TopKSampler,
        'hybrid': HybridSampler
    }

    if strategy.lower() not in samplers:
        raise ValueError(f"Unknown sampling strategy: {strategy}. "
                        f"Available: {list(samplers.keys())}")

    return samplers[strategy.lower()](**kwargs)

