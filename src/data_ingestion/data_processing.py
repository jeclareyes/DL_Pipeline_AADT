"""
Descripción
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union, Tuple, Dict
from scipy import sparse
import sys
import numpy as np

# Importaciones de módulos 'data_processing'
from processing_modules._data_loader import DataLoader

# Importaciones de Hydra para gestión de configuraciones
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

# Definición de PROYECT_ROOT para que siempre se ejecute desde la raíz del proyecto

class DataManager:
    """
    Gestor centralizado de datos para Barcelona GNN (generalizado para múltiples redes).

    Mantengo el nombre de la clase `DataManager` para compatibilidad con código
    existente; el módulo ahora se llama `data_processing`.
    """

    def __init__(self,
                 cfg: Optional[DictConfig] = None):
        """
        If cfg is provided, it overrides individual path parameters.
        """

        #%% Cargando configuración desde Hydra

        self.config = cfg if isinstance(cfg, DictConfig) else None

        if self.config is not None:
            # resolve data root via Hydra helper (makes paths absolute regardless of Hydra cwd)
            try:
                resolved_route = to_absolute_path(self.config.input_routes.general_route)
                data_root = Path(resolved_route)
            except Exception:
                data_root = Path(self.config.input_routes.general_route) if self.config.input_routes.general_route else None
        else:
            print("Error")

        #%% Inicializando atributos básicos y rutas

        self.network_name = self.config.dataset
        self.multiday = self.config.multiday_od
        self.volume_year = self.config.volume_year

        # metadata container (initialize early)
        self.metadata = {
            'network_name': self.network_name,
            'multiday': self.multiday
        }

        input_route = Path(self.config.input_routes.general_route) if self.config.input_routes.general_route else None

        self.flow_route =       Path(to_absolute_path(self.config.input_routes.flow_route))
        self.network_route =    Path(to_absolute_path(self.config.input_routes.network_route))
        self.node_route =       Path(to_absolute_path(self.config.input_routes.node_route))
        self.routes_route =     Path(to_absolute_path(self.config.input_routes.routes_route))
        self.trips_route =      Path(to_absolute_path(self.config.input_routes.trips_route))

        print(self.flow_route)

        self.all_routes = {
            'flow_route': self.flow_route,
            'network_route': self.network_route,
            'node_route': self.node_route,
            'routes_route': self.routes_route,
            'trips_route': self.trips_route
        }

        #%% Métdo core de carga de datos
        from src.data_ingestion.processing_modules._data_loader import DataLoader
        # DataLoader will update this manager's attributes (network_df, flow_df, od_matrix, node_coords_df, metadata)
        loader  = DataLoader(routes=self.all_routes,
                             volume_year=self.volume_year,
                             multiday=self.multiday,
                             run_loads=True)

        # Read results from loader attributes
        network_df = loader.network_df
        flow_df = loader.flow_df
        known_od_matrix = loader.od_matrix
        node_df = loader.node_df

        #%% Expansión de Matriz OD para inclusión de pares OD desconocidos

        from src.data_ingestion.processing_modules._od_matrix_expander import add_aux_od_matrix

        complete_od_matrix = add_aux_od_matrix(known_od_matrix, node_df)

        #%% Merging de network y flow dataframes (esto para tener toda la data de links en un solo dataframe)

        from src.data_ingestion.processing_modules._unify_link_based_data import unify_link_data

        link_df = unify_link_data(network_df, flow_df)

        #%% Construcción de grafo (NetworkX) - como elemento base de todos los problemas de estimación.
        # Además, se usa para el cálculo de rutas k-shortest paths.

        from src.data_ingestion.processing_modules._graph_network_creator import build_graph
        graph = build_graph(link_df, node_df)

        #%% Computación de rutas k-shortest paths (a partir del grafo que hemos construido)

        from src.data_ingestion.processing_modules._process_routes import RouteHandler

        route_handler = RouteHandler(
            graph=graph,
            node_df=node_df,
            output_route = self.config.output_routes.routing_cache_route,
            config=self.config.route_calculation)

        routes_data = route_handler.run()

        #%% Save processed data

        from src.data_ingestion.processing_modules._data_saver import DataSaver
        saver = DataSaver(processed_dir=self.config.output_routes.processed_route)

        # 1. Guardar grafo (pickle y opcionalmente imagen)
        saver.save_graph_pickle(graph, file_path=Path(self.config.output_routes.processed_route) / f"{self.network_name}_graph.pkl")
        # saver.save_graph_image(graph, file_path=Path(self.config.output_routes.processed_route) / f"{self.network_name}_graph.png", fmt='png')

        # 2. Guardar matriz OD (formato sparse .npz)
        # TODO meejorar este guardado para que sea más eficiente
        from scipy.sparse import csr_matrix
        complete_od_matrix = csr_matrix(complete_od_matrix)
        od_matrix_path = Path(self.config.output_routes.processed_route) / f"{self.network_name}_od_matrix.npz"
        sparse.save_npz(od_matrix_path, complete_od_matrix)

        # 3. Guardar dataframe unificado de links (parquet)
        link_data_path = Path(self.config.output_routes.processed_route) / f"{self.network_name}_link_data.parquet"
        link_df.to_parquet(link_data_path, index=False)


@hydra.main(config_path="../../configs/data_ingestion/data_processing", config_name="data_processing")
def main(cfg):
    dm = DataManager(cfg=cfg)
    try:
        print("--- Cargando red de tráfico ---")
        # dm.load_network()
        # dm.load_flow()
        # dm.load_od_matrix()
        # dm.merge_network_flow()
    except Exception as e:
        print(f"Error during data processing: {e}")
        raise

if __name__ == '__main__':
    main()