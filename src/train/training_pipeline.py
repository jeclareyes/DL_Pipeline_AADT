import logging

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

if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def run_pipeline(cfg: DictConfig):
    print(f"[HYDRA] Iniciando Pipeline. Modelo: {cfg.model._target_}")
    device = cfg.training.device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    # --- 1. DATA INGESTION (Usando tu clase adaptada) ---
    print("Cargando datos estructurales...")
    # config entera. El loader sacará las rutas de ahí.
    loader = LinkopingDataLoader(cfg)

    loader.load_all()
    # tensores estructurales (t0, capacity, masks)
    # Method prepare_network_parameters() .
    network_params = loader.prepare_network_parameters()

    all_flows, train_mask, test_mask = loader.prepare_observed_flows()
    od_vector, od_mask = loader.prepare_od_demand_vector()

    # --- 2. SAMPLING (Integrando engine) ---
    print("🎲 Ejecutando Sampling Engine...")
    sampling_engine = SamplingEngine(config=cfg)

    # (Ajusta esto según el retorno real de engine.run())
    sampled_flow_mask = sampling_engine.run(save=False)

    # Si engine no retorna la máscara OD, generamos o sacamos del loader
    sampled_od_mask = od_mask  # TODO hacer sampling de OD

    # --- 3. INSTANCIACIÓN DEL MODELO  ---
    print("⚙️ Construyendo Modelo Dinámicamente...")

    # Hydra toma los hiperparámetros del YAML (hidden_dim, etc.)
    # Inyección de tensores
    logging.info("Instanciando modelo con Hydra...")
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
    # Convertir a tensores y mover DIRECTAMENTE al device (GPU/CPU)

    logging.info("Enviando tensores de flujos y pares OD al device...")
    true_flows_t = torch.FloatTensor(all_flows).to(device)
    true_od_t = torch.FloatTensor(od_vector).to(device)
    flow_mask_t = torch.FloatTensor(sampled_flow_mask).to(device)
    od_mask_t = torch.FloatTensor(sampled_od_mask).to(device)

    logging.info("Configurando optimizador y función de pérdida...")
    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    # Loss weights desde el YAML del modelo
    criterion = PartialDataLoss(
        w_flow=cfg.model.loss_weights.w_flow,
        w_od=cfg.model.loss_weights.w_od,
        w_reg=cfg.model.loss_weights.w_reg
    ).to(device)

    # --- 5. LOOP DE ENTRENAMIENTO ---
    logging.info("Iniciando loop de entrenamiento (Full Batch)...")
    model.train()

    min_iters = 2
    max_iters_target = cfg.model.get('max_iters', 10)

    for epoch in range(cfg.training.epochs):
        # PASAMOS TODOS LOS DATOS DE GOLPE
        # Nota: Asegúrate de que tu modelo acepte las dimensiones sin la dimensión extra del batch
        # O añade una dimensión 'falsa' de batch si el modelo lo espera: .unsqueeze(0)

        """# Cálculo dinámico de iteraciones
        # Subimos 1 iteración cada 5 épocas, por ejemplo
        if epoch < 5:
            current_sue_iters = 1  # Fase inicial muy rápida
        else:
            # Crecimiento progresivo
            growth = (epoch - 5) // 5
            current_sue_iters = min(min_iters + growth, max_iters_target)

        # Imprimir para control
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | SUE Iters: {current_sue_iters}")"""


        optimizer.zero_grad()

        # Forward
        outputs = model(
            observed_flows=true_flows_t,
            flow_mask=flow_mask_t,
            warmup=(epoch < 5)
        )

        # Loss
        loss_dict = criterion(
            predicted_flows=outputs['reconstructed_flows'],
            true_flows=true_flows_t,
            flow_mask=flow_mask_t,
            predicted_od=outputs['estimated_demand'],
            true_od=true_od_t,
            od_mask=od_mask_t,
            learned_alpha=outputs.get('learned_alpha'),
            learned_beta=outputs.get('learned_beta')
        )

        loss_dict['total_loss'].backward()
        optimizer.step()

        if epoch % 10 == 0:
            logging.info(f"""Epoch {epoch}: Loss {loss_dict['total_loss'].item():.4f} - Flow Loss {loss_dict['l_flow'].item():.4f} - OD Loss {loss_dict['l_od'].item():.4f} - Reg Loss {loss_dict['l_reg'].item():.4f}""")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    run_pipeline(cfg)


if __name__ == "__main__":
    main()