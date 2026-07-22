import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

def build_route_flows_table(od_indices: np.ndarray, route_flows: np.ndarray, route_validity_mask: np.ndarray, estimated_od: np.ndarray, delta_matrix: np.ndarray) -> pd.DataFrame:
    """Builds a DataFrame detailing flows for each valid route."""
    num_od_pairs, max_routes = route_flows.shape
    
    if len(delta_matrix.shape) == 2:
        num_links, total_routes = delta_matrix.shape
        delta_matrix = delta_matrix.reshape((num_links, num_od_pairs, max_routes))
    
    rows = []
    for od_idx in range(num_od_pairs):
        origin = od_indices[od_idx, 0]
        destination = od_indices[od_idx, 1]
        od_flow = estimated_od[od_idx]
        
        for route_idx in range(max_routes):
            is_valid = bool(route_validity_mask[od_idx, route_idx])
            if not is_valid:
                continue
                
            r_flow = route_flows[od_idx, route_idx]
            r_share = r_flow / od_flow if od_flow > 0 else 0
            
            # Route length in links (sum of delta matrix for this route)
            r_length = np.sum(delta_matrix[:, od_idx, route_idx]) if delta_matrix is not None else 0
            
            rows.append({
                "od_idx": od_idx,
                "origin": origin,
                "destination": destination,
                "route_idx": route_idx,
                "route_valid": is_valid,
                "route_flow": float(r_flow),
                "route_share_within_od": float(r_share),
                "estimated_od": float(od_flow),
                "route_length_links": int(r_length)
            })
            
    return pd.DataFrame(rows)

def build_route_link_contributions(od_indices: np.ndarray, route_flows: np.ndarray, route_validity_mask: np.ndarray, delta_matrix: np.ndarray, estimated_flows: np.ndarray) -> pd.DataFrame:
    """Builds a DataFrame detailing how each route contributes to each link."""
    num_od_pairs, max_routes = route_flows.shape
    if len(delta_matrix.shape) == 2:
        num_links, total_routes = delta_matrix.shape
        delta_matrix = delta_matrix.reshape((num_links, num_od_pairs, max_routes))
    
    num_links, _, _ = delta_matrix.shape
    
    rows = []
    # Find all (link, od, route) triplets where delta == 1 and route is valid
    link_indices, od_indices_arr, route_indices = np.where(delta_matrix > 0)
    
    for l_idx, o_idx, r_idx in zip(link_indices, od_indices_arr, route_indices):
        if not route_validity_mask[o_idx, r_idx]:
            continue
            
        origin = od_indices[o_idx, 0]
        destination = od_indices[o_idx, 1]
        r_flow = route_flows[o_idx, r_idx]
        link_flow = estimated_flows[l_idx]
        
        od_flow = np.sum(route_flows[o_idx, route_validity_mask[o_idx].astype(bool)])
        r_share_od = r_flow / od_flow if od_flow > 0 else 0
        r_share_link = r_flow / link_flow if link_flow > 0 else 0
        
        rows.append({
            "link_index": l_idx,
            "origin": origin,
            "destination": destination,
            "od_idx": o_idx,
            "route_idx": r_idx,
            "route_flow": float(r_flow),
            "route_share_within_link": float(r_share_link),
            "route_share_within_od": float(r_share_od)
        })
        
    return pd.DataFrame(rows)

def build_od_link_contributions(od_indices: np.ndarray, route_flows: np.ndarray, route_validity_mask: np.ndarray, delta_matrix: np.ndarray, estimated_flows: np.ndarray) -> pd.DataFrame:
    """Builds a DataFrame detailing how each OD pair contributes to each link."""
    num_od_pairs, max_routes = route_flows.shape
    if len(delta_matrix.shape) == 2:
        num_links, total_routes = delta_matrix.shape
        delta_matrix = delta_matrix.reshape((num_links, num_od_pairs, max_routes))
        
    num_links, _, _ = delta_matrix.shape
    
    rows = []
    
    # Calculate OD flow per link
    # For each link, flow from OD = sum(route_flows for valid routes passing through link)
    for l_idx in range(num_links):
        link_flow = estimated_flows[l_idx]
        if link_flow == 0:
            continue
            
        for o_idx in range(num_od_pairs):
            # Sum of flows for valid routes of this OD pair passing through this link
            valid_routes_on_link = (delta_matrix[l_idx, o_idx] > 0) & (route_validity_mask[o_idx] > 0)
            if not np.any(valid_routes_on_link):
                continue
                
            od_flow_on_link = np.sum(route_flows[o_idx, valid_routes_on_link])
            if od_flow_on_link == 0:
                continue
                
            share_within_link = od_flow_on_link / link_flow
            
            rows.append({
                "link_index": l_idx,
                "origin": od_indices[o_idx, 0],
                "destination": od_indices[o_idx, 1],
                "od_idx": o_idx,
                "total_od_flow_through_link": float(od_flow_on_link),
                "share_within_link": float(share_within_link)
            })
            
    return pd.DataFrame(rows)
