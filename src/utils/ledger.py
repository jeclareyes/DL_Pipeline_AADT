import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.utils.paths import get_project_root, resolve_path

def update_experiments_ledger(
    cfg: DictConfig,
    run_hash: str,
    output_csv: str = "outputs/experiments_ledger.csv",
):
    """
    Guarda la configuración del experimento (la config de Hydra) aplanada en un CSV como base de datos ligera.
    """
    csv_path = resolve_path(output_csv, relative_to=get_project_root())
    
    # Flatten the configuration dict (using pandas json_normalize)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # We filter out redundant/huge fields if needed, or select target ones:
    target_configs = {}
    for key in ['dataset', 'model', 'training', 'sampling', 'vdf']:
        if key in config_dict:
            target_configs[key] = config_dict[key]
            
    flat_config = pd.json_normalize(target_configs, sep='_').to_dict(orient='records')[0]
    
    # Add metadata
    flat_config['run_hash'] = run_hash
    flat_config['model_name'] = cfg.model.get('model_name', 'Unknown')
    
    # Create DataFrame from flattened dict
    df_new = pd.DataFrame([flat_config])
    
    # Append or create
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        # Avoid duplicates for same run_hash just in case, though UUID makes it nearly impossible
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
    else:
        # Create output dir if needed
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(csv_path, index=False)
