import logging

import hydra
import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

# Importaciones de TU estructura
# from src.components.models.Cyclic_Model.cyclic_model import PartialDataLoss
from src.components.models.Cyclic_Model.cyclic_model_data_ingestion import LinkopingDataLoader
# Asumiendo que moviste el dataset a utils, si no, impórtalo de train_cyclic_model
from src.utils.traffic_dataset import TrafficDataset
# Tu motor de sampling existente
from src.components.sampling.engine import SamplingEngine

from src.train._saving_handler import save_checkpoint

import os
import sys
import io

if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def run_pipeline(cfg: DictConfig):
    # Hydra ya ha cambiado el directorio de trabajo a la carpeta de salida configurada en config.yaml
    output_dir = cfg.runs.dir
    logging.info(f"Directorio de salida del run: {output_dir}")

    # ---------------------------------------------------------
    # 1. CONSTRUCCIÓN DEL NOMBRE DEL ARCHIVO (Largo y descriptivo)
    # ---------------------------------------------------------
    # Usamos getattr o get para evitar errores si alguna clave no existe
    # Asumimos que 'network.cost_function' se refiere a cfg.network.cost_function
    # Si tu config usa 'vdf' en lugar de 'network', ajusta abajo (ej. cfg.vdf.name)

    base_name = (
        f"Epochs_{cfg.training.epochs}_"
        f"VDF_{cfg.get('network', {}).get('cost_function', 'UnknownVDF')}_"
        f"Learning_Rate_{cfg.training.lr}_"
        f"Flow_Sampling_Rate_{cfg.sampling.flow_rate}_"
        f"Sampling_Strategy_{cfg.sampling.strategy}_"
        f"Sampling_Basis_{cfg.sampling.sampling_basis}"
    )

    model_filename = f"{base_name}.pt"
    eval_filename = (f"eval_{base_name}.pt")

    full_model_path = os.path.join(output_dir, model_filename)
    full_eval_path = os.path.join(output_dir, eval_filename)

    logging.info(f"El entrenamiento se guardará en: {model_filename}")
    logging.info(f"Las evaluaciones se guardarán en: {eval_filename}")

    # ---------------------------------------------------------
    # 2. INICIO DEL PIPELINE
    # ---------------------------------------------------------

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
    #criterion = PartialDataLoss(
    #    w_flow=cfg.model.loss_weights.w_flow,
    #    w_od=cfg.model.loss_weights.w_od,
    #    w_reg=cfg.model.loss_weights.w_reg
    #).to(device)

    # --- CAMBIO IMPORTANTE: Instanciación dinámica del Loss ---
    if hasattr(cfg.model, 'loss') and '_target_' in cfg.model.loss:
        # Caso: Modelo Ultra con configuración de Loss propia
        logging.info(f"Instanciando Loss desde config: {cfg.model.loss._target_}")
        criterion = hydra.utils.instantiate(cfg.model.loss).to(device)
    else:
        # Fallback: Modelo Clásico (Cyclic_Model.yaml original)
        # Importación local para evitar errores si no se usa
        from src.components.models.Cyclic_Model.cyclic_model import PartialDataLoss
        logging.info("Usando PartialDataLoss por defecto (Legacy)")
        criterion = PartialDataLoss(
            w_flow=cfg.model.loss_weights.w_flow,
            w_od=cfg.model.loss_weights.w_od,
            w_reg=cfg.model.loss_weights.w_reg
        ).to(device)

    # Preparar estructura de guardado maestro

    # Inicializamos el diccionario maestro que contendrá TODAS las epochs
    master_checkpoint = {
        'config': OmegaConf.to_container(cfg, resolve=True),  # Guardamos config globalmente
        'epochs_history': {}  # Aquí acumularemos cada checkpoint
    }

    master_eval_bundle = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'static_data': {
            # Guardamos esto una sola vez porque es estático
            'capacity': network_params['capacity'].cpu(),
            't0': network_params['t0'].cpu(),
            'link_group': network_params['link_group'].cpu(),
            # Inputs / Ground Truth
            'true_flows': torch.FloatTensor(all_flows).cpu(),
            'true_od': torch.FloatTensor(od_vector).cpu(),
            'mask_flow_train': torch.BoolTensor(train_mask).cpu(),
            'mask_flow_test': torch.BoolTensor(test_mask).cpu(),
            'mask_od_known': torch.BoolTensor(od_mask).cpu(),  # Si aplica
        },
        'epochs_history': {}  # Aquí guardaremos las predicciones evolutivas
    }


    # --- 5. LOOP DE ENTRENAMIENTO ---
    logging.info("Iniciando loop de entrenamiento (Full Batch)...")
    model.train()

    min_iters = 2
    max_iters_target = cfg.model.get('max_iters', 10)

    for epoch in range(cfg.training.epochs):
        # PASAMOS TODOS LOS DATOS DE GOLPE
        # Nota: Asegúrate de que tu modelo acepte las dimensiones sin la dimensión extra del batch
        # O añade una dimensión 'falsa' de batch si el modelo lo espera: .unsqueeze(0)

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

        current_loss = loss_dict['total_loss'].item()

        # ---------------------------------------------------------
        # 2. GUARDADO ACUMULATIVO EN UN SOLO ARCHIVO
        # ---------------------------------------------------------
        # Guardamos cada 10 epochs Y TAMBIÉN la primer y última
        if (epoch + 1) == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) == cfg.training.epochs:
            # 1. Crear el estado de la epoch actual
            epoch_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
                # CORRECCIÓN: Verificamos si v es Tensor antes de llamar a .item()
                'metrics': {
                    k: (v.item() if isinstance(v, torch.Tensor) else v)
                    for k, v in loss_dict.items()
                }
            }

            # 2. Añadirlo al historial del maestro usando el número de epoch como clave
            master_checkpoint['epochs_history'][epoch + 1] = epoch_state

            # 3. Sobreescribir el archivo único con el historial actualizado
            torch.save(master_checkpoint, full_model_path)

            # --- 2. Guardar Datos de Evaluación (Checkpoint pesado) ---
            # Extraemos los datos del último forward pass
            eval_snapshot = {
                'pred_flows': outputs['reconstructed_flows'].detach().cpu(),
                'pred_od': outputs['estimated_demand'].detach().cpu(),
                'route_probs': outputs['route_probs'].detach().cpu(),  # <--- Aquí van las probs
                'alpha': outputs.get('learned_alpha', torch.tensor(-1)).detach().cpu(),
                'beta': outputs.get('learned_beta', torch.tensor(-1)).detach().cpu(),
                'convergence': outputs.get('convergence_info', {})
            }

            master_eval_bundle['epochs_history'][epoch + 1] = eval_snapshot
            torch.save(master_eval_bundle, full_eval_path)

            logging.info(f"[Epoch {epoch + 1}] Checkpoint y Evaluación actualizados.")

    logging.info("Entrenamiento completado.")

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    run_pipeline(cfg)


if __name__ == "__main__":
    main()