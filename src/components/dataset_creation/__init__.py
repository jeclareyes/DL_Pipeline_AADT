from hydra.core.config_store import ConfigStore
from .config import DatasetConfig
from .orchestration import dataset_creation_orchestration

cs = ConfigStore.instance()
cs.store(group="dataset_creation", name="base_config", node=DatasetConfig)

__all__ = ["dataset_creation_orchestration"]
