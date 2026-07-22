from dataclasses import dataclass, field
from typing import Dict, Mapping, Type
from .base import BaseCostFunction
from .bpr import BPRCostFunction
from .conical import ConicalCostFunction
from .akcelik import AkcelikCostFunction

VDF_REGISTRY: Dict[str, Type[BaseCostFunction]] = {
    "bpr": BPRCostFunction,
    "conical": ConicalCostFunction,
    "akcelik": AkcelikCostFunction
}

@dataclass(frozen=True)
class VDFConfig:
    """
    Configuración central para cualquier Volume-Delay Function.
    Reemplaza a BPRCostFunctionConfig.
    """
    vdf_name: str
    free_flow_time_col: str
    capacity_col: str
    toll_col: str | None
    toll_required: bool
    parameters_cols: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.vdf_name:
            raise ValueError("VDFConfig requires a valid vdf_name.")
        if self.vdf_name.lower() not in VDF_REGISTRY:
            raise ValueError(f"Unsupported VDF '{self.vdf_name}'. Available: {list(VDF_REGISTRY.keys())}")
        if not self.free_flow_time_col:
            raise ValueError("VDFConfig requires free_flow_time_col.")
        if not self.capacity_col:
            raise ValueError("VDFConfig requires capacity_col.")
        if self.toll_required and not self.toll_col:
            raise ValueError("VDFConfig requires toll_col if toll_required is True.")
        if not isinstance(self.parameters_cols, Mapping):
            raise TypeError("VDFConfig.parameters_cols must be a mapping.")
        for key, value in self.parameters_cols.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"VDFConfig.parameters_cols contains an invalid key: {key!r}")
            if not isinstance(value, str) or not value:
                raise ValueError(f"VDFConfig.parameters_cols[{key!r}] must be a non-empty string.")

    def get_vdf_class(self) -> Type[BaseCostFunction]:
        return VDF_REGISTRY[self.vdf_name.lower()]

    def get_columns_kwargs(self) -> dict:
        """
        Retorna el diccionario de columnas mapeadas para inyectar en evaluate_costs_numpy.
        Combina las columnas base con los parámetros específicos de la VDF.
        """
        kwargs = {
            "free_flow_time_col": self.free_flow_time_col,
            "capacity_col": self.capacity_col,
            "toll_col": self.toll_col,
        }
        kwargs.update(self.parameters_cols)
        return kwargs
