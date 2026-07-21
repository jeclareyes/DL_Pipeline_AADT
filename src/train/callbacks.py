from hydra.experimental.callback import Callback
from omegaconf import DictConfig
import os
import logging
import sys
from typing import Any

# Hay que ajustar el path para poder importar scripts si no está en PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.generate_optuna_report import generate_report

log = logging.getLogger(__name__)

class OptunaReportCallback(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Se ejecuta al finalizar todos los jobs del multirun.
        """
        log.info("OptunaReportCallback: on_multirun_end triggered.")
        
        # Obtenemos el directorio donde hydra guardó los resultados del sweep
        # En versiones recientes de Hydra, config.hydra.sweep.dir es confiable
        sweep_dir = config.hydra.sweep.dir if config.hydra.sweep.dir else "."
            
        # Resolver ruta absoluta
        # Nota: Hydra cambia el cwd, así que usamos get_original_cwd() si está disponible,
        # o confiamos en rutas absolutas.
        try:
             import hydra.utils
             base_dir = hydra.utils.get_original_cwd()
             # Si sweep_dir es relativo, lo unimos al original cwd
             if not os.path.isabs(sweep_dir):
                 abs_sweep_dir = os.path.join(base_dir, sweep_dir)
             else:
                 abs_sweep_dir = sweep_dir
        except Exception:
             # Fallback si falla get_original_cwd
             abs_sweep_dir = os.path.abspath(sweep_dir)
            
        log.info(f"Finalizando Multirun. Generando reporte de Optuna en: {abs_sweep_dir}")
        
        try:
            generate_report(abs_sweep_dir)
        except Exception as e:
            log.error(f"Error generando reporte automático de Optuna: {e}")
            import traceback
            log.error(traceback.format_exc())
