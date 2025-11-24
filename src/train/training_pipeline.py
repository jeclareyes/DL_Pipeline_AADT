import hydra
import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import DictConfig

# Importaciones de TU estructura
from src.components.models.Cyclic_Model.cyclic_model import PartialDataLoss
from src.components.models.Cyclic_Model.cyclic_model_data_ingestion import LinkopingDataLoader
# Asumiendo que moviste el dataset a utils, si no, impórtalo de train_cyclic_model
from src.utils.traffic_dataset import TrafficDataset
# Tu motor de sampling existente
from src.components.sampling.engine import SamplingEngine

import os
import sys
import io

# Parche para Windows: Forzar salida UTF-8 para soportar emojis en logs
if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def run_pipeline(cfg: DictConfig):
    print(f"🚀 [HYDRA] Iniciando Pipeline. Modelo: {cfg.model._target_}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. DATA INGESTION (Usando tu clase adaptada) ---
    print("📊 Cargando datos estructurales...")
    # Pasamos la config entera. El loader sacará las rutas de ahí.
    loader = LinkopingDataLoader(cfg)

    loader.load_all()
    # Obtenemos los tensores estructurales (t0, capacity, masks)
    # Tu método prepare_network_parameters() es CLAVE aquí.
    network_params = loader.prepare_network_parameters()

    all_flows, train_mask, test_mask = loader.prepare_observed_flows()
    od_vector, od_mask = loader.prepare_od_demand_vector()

    # --- 2. SAMPLING (Integrando tu engine) ---
    print("🎲 Ejecutando Sampling Engine...")
    # Tu SamplingEngine parece esperar un config dict.
    # Hydra permite convertir su config a dict nativo si es necesario.
    # Si tu engine usa internamente las claves de 'sampling.yaml', funcionará.
    sampling_engine = SamplingEngine(config=cfg)

    # Supongamos que tu engine devuelve las máscaras muestreadas
    # (Ajusta esto según el retorno real de tu engine.run())
    sampled_flow_mask = sampling_engine.run(save=False)

    # Si tu engine no retorna la máscara OD, la generamos o la sacamos del loader
    sampled_od_mask = od_mask  # Placeholder si no haces sampling de OD aún

    # --- 3. INSTANCIACIÓN DEL MODELO (El momento mágico) ---
    print("⚙️ Construyendo Modelo Dinámicamente...")

    # Hydra toma los hiperparámetros del YAML (hidden_dim, etc.)
    # Nosotros le "inyectamos" los tensores pesados que acabamos de cargar.
    model = hydra.utils.instantiate(
        cfg.model,
        # Argumentos dinámicos (**kwargs)
        num_links=network_params['num_links'],
        num_od_pairs=network_params['num_od_pairs'],
        t0=network_params['t0'].to(device),
        capacity=network_params['capacity'].to(device),
        route_masks=network_params['route_masks'].to(device),
        od_pair_indices=network_params['od_pair_indices'].to(device),
        num_link_groups=network_params['num_link_groups'],
        link_group=network_params['link_group'].to(device),
        _recursive_=False
    )
    model.to(device)

    # --- 4. PREPARACIÓN DE ENTRENAMIENTO ---
    # Convertir a tensores para el DataLoader
    true_flows_t = torch.FloatTensor(all_flows).unsqueeze(0)
    true_od_t = torch.FloatTensor(od_vector).unsqueeze(0)
    flow_mask_t = torch.FloatTensor(sampled_flow_mask).unsqueeze(0)
    od_mask_t = torch.FloatTensor(sampled_od_mask).unsqueeze(0)

    dataset = TrafficDataset(
        true_flows_t.numpy(), true_od_t.numpy(),
        flow_mask_t.numpy(), od_mask_t.numpy()
    )
    train_loader = DataLoader(dataset, batch_size=cfg.training.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    # Loss weights desde el YAML del modelo
    criterion = PartialDataLoss(
        w_flow=cfg.model.loss_weights.w_flow,
        w_od=cfg.model.loss_weights.w_od,
        w_reg=cfg.model.loss_weights.w_reg
    ).to(device)

    # --- 5. LOOP DE ENTRENAMIENTO ---
    print("🔥 Iniciando Epochs...")
    model.train()
    for epoch in range(cfg.training.epochs):
        for batch in train_loader:
            # Desempaquetar batch (ajusta según tu TrafficDataset)
            b_flows, b_od, b_flow_mask, b_od_mask = batch

            # Mover a GPU
            b_flows = b_flows.to(device)
            b_flow_mask = b_flow_mask.to(device)

            optimizer.zero_grad()

            # Forward
            outputs = model(
                observed_flows=b_flows,
                flow_mask=b_flow_mask,
                warmup=(epoch < 5)
            )

            # Loss
            loss_dict = criterion(
                predicted_flows=outputs['reconstructed_flows'],
                true_flows=b_flows,
                flow_mask=b_flow_mask,
                # ... pasa el resto de argumentos a tu loss ...
                learned_alpha=outputs.get('learned_alpha'),
                learned_beta=outputs.get('learned_beta')
            )

            loss_dict['total_loss'].backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss_dict['total_loss'].item():.4f}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    run_pipeline(cfg)


if __name__ == "__main__":
    main()