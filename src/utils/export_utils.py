import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely import wkb, wkt

logger = logging.getLogger(__name__)

def export_to_geopackage(
    raw_data: dict,
    reconstructed_flows: np.ndarray,
    estimated_demand: np.ndarray,
    output_dir: str | Path,
    filename: str = "model_results.gpkg"
):
    """
    Exporta las estimaciones del modelo a un archivo GeoPackage (.gpkg).
    Asume que la geometría origen de raw_data['link_data'] existe en crudo.
    """
    gpkg_path = Path(output_dir) / filename
    graph = raw_data['graph']
    link_data = raw_data['link_data'].copy()
    
    # -------------------------------------------------------------
    # 1. Capa de Enlaces (Links Estimations)
    # -------------------------------------------------------------
    # Aseguramos de que el link_data y flujos tengan el mismo orden temporal
    if len(link_data) != len(reconstructed_flows):
        logger.warning(f"Tamaño de link_data ({len(link_data)}) no coincide con predicciones ({len(reconstructed_flows)}).")
    
    # Anexar Predicciones
    link_data['estimated_flow'] = reconstructed_flows.squeeze()
    if 'volume' in link_data.columns:
        link_data['ground_truth_flow'] = link_data['volume']
        link_data['error_absoluto'] = np.abs(link_data['estimated_flow'] - link_data['ground_truth_flow'])
    else:
        link_data['error_absoluto'] = np.nan
        
    links_gdf = pd.DataFrame(link_data)
    
    # Extraer geometrías originales (WKB u objetos) o degradar a rectas entre nodos 
    if 'geometry' in link_data.columns:
        geometries = []
        for geom in link_data['geometry']:
            if isinstance(geom, (bytes, bytearray)):
                try: geometries.append(wkb.loads(geom))
                except Exception: geometries.append(None)
            elif isinstance(geom, str):
                try: geometries.append(wkt.loads(geom))
                except Exception: geometries.append(None)
            else:
                geometries.append(geom)

        links_gdf['geometry'] = geometries
        links_gdf = gpd.GeoDataFrame(links_gdf, geometry='geometry')
        # Es prudente asignar el CRS por defecto si no lo lleva (generalmente WGS84 para mapas o proyectados 3857)
        if links_gdf.crs is None: 
            links_gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
    else:
        logger.warning("Columna geometry NO encontrada en link_data. Construyendo líneas rectas como fallback.")
        edges = list(graph.edges(data=True))
        node_coords = {n: (d.get('x', 0), d.get('y', 0)) for n, d in graph.nodes(data=True)}
        
        init_col = 'init_node' if 'init_node' in link_data.columns else 'from_node'
        term_col = 'term_node' if 'term_node' in link_data.columns else 'to_node'
        
        geoms = []
        for _, row in link_data.iterrows():
            if init_col in row and term_col in row:
                u, v = row[init_col], row[term_col]
                if u in node_coords and v in node_coords:
                    geoms.append(LineString([node_coords[u], node_coords[v]]))
                else: geoms.append(None)
            else: geoms.append(None)
        
        links_gdf['geometry'] = geoms
        links_gdf = gpd.GeoDataFrame(links_gdf, geometry='geometry')

    links_gdf = links_gdf.dropna(subset=['geometry'])
    # Se exporta capa 1 (Links Viales)    
    links_gdf.to_file(str(gpkg_path), layer='links_estimations', driver='GPKG', engine='pyogrio')
    
    # -------------------------------------------------------------
    # 2. Capa de Demanda OD (Líneas de Deseo / Desire Lines)
    # -------------------------------------------------------------
    if 'od_matrix' in raw_data and estimated_demand is not None:
        try:
            true_od_dense = raw_data['od_matrix'].toarray()
        except AttributeError:
            true_od_dense = np.array(raw_data['od_matrix'])
            
        # Ensure it has 2 dimensions before unpacking
        if estimated_demand.ndim == 2:
            num_origins, num_destinations = estimated_demand.shape
            od_rows = []
            node_coords = {n: (d.get('x', 0), d.get('y', 0)) for n, d in graph.nodes(data=True)}
            nodes_list = sorted(list(graph.nodes()))
                
        for i in range(num_origins):
            for j in range(num_destinations):
                o_node = nodes_list[i] if i < len(nodes_list) else None
                d_node = nodes_list[j] if j < len(nodes_list) else None
                
                est_val = estimated_demand[i, j]
                true_val = true_od_dense[i, j] if i < true_od_dense.shape[0] and j < true_od_dense.shape[1] else 0.0
                
                if est_val > 0.01 or true_val > 0.01:
                    geom = None
                    if o_node in node_coords and d_node in node_coords:
                        geom = LineString([node_coords[o_node], node_coords[d_node]])
                    
                    od_rows.append({
                        'origin_node': o_node,
                        'destination_node': d_node,
                        'estimated_demand': float(est_val),
                        'ground_truth_demand': float(true_val),
                        'error_absoluto': float(abs(est_val - true_val)),
                        'geometry': geom
                    })
                    
        od_gdf = gpd.GeoDataFrame(od_rows)
        od_gdf = od_gdf.dropna(subset=['geometry'])
        
        if not od_gdf.empty:
            if links_gdf.crs is not None:
                od_gdf.set_crs(links_gdf.crs, inplace=True)
            # Se exporta capa 2 (Desire Lines)
            od_gdf.to_file(str(gpkg_path), layer='od_estimations', driver='GPKG', engine='pyogrio')
        else:
            logger.warning("GeoDataFrame OD vacío tras limpiar, no se exportó la capa OD.")
            
    logger.info("Resultados exitosamente exportados a: %s", gpkg_path)
