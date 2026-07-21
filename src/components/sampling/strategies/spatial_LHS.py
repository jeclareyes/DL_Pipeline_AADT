import logging
import numpy as np
import networkx as nx
from typing import Optional, List, Set
from scipy.stats import qmc
from scipy.spatial import cKDTree

from src.components.sampling.base import BaseSampler
from src.components.sampling import sampling as utils

# Logger specific for this strategy
logger = logging.getLogger(__name__)


class SpatialLHSStrategy(BaseSampler):
    """
    Implements Latin Hypercube Sampling (LHS) considering spatial distribution
    and link types.

    Includes a configurable 'Random Fill' fallback to ensure the target sample size
    is met when high sampling rates cause KDTree collisions.
    """

    def create_partial_data_masks(
            self,
            train_flow_mask: np.ndarray,
            od_mask: np.ndarray,
            flow_rate: float,
            od_rate: float,
            graph: nx.Graph,
            volume_year: Optional[int] = None
    ):
        # 1. Configuration & Parameters
        seed = self.params.get("random_seed", 42)
        basis = self.params.get("sampling_basis", "traffic_counts_based")

        # New parameter from YAML to toggle the filling behavior
        # Default to True to maintain robust data volume unless specified otherwise
        enable_random_fill = self.params.get("random_fill", True)

        # 2. Extract Data & Define Universe (Groups)
        internal_df = utils.extract_graph_data(graph, volume_year)
        groups = utils.get_observation_groups(train_flow_mask, basis, internal_df)

        # --- LOGGING: Redundancy Analysis ---
        n_raw_links = int(train_flow_mask.sum())
        n_groups = len(groups)

        if basis == "traffic_counts_based":
            logger.info(f"[Grouping Analysis] Strategy: Spatial LHS | Basis: {basis}")
            logger.info(f"   - Raw Active Links: {n_raw_links}")
            logger.info(f"   - Consolidated Groups (Sensors): {n_groups}")
            if n_raw_links > 0:
                reduction = 100 * (1 - n_groups / n_raw_links)
                logger.info(f"   - Redundancy Removed: {reduction:.1f}%")
        else:
            logger.info(f"[Grouping Analysis] Strategy: Spatial LHS | Basis: {basis}")

        # 3. Target Calculation
        num_samples_target = int(n_groups * flow_rate)
        selected_groups = []
        selected_indices_set: Set[int] = set()

        # 4. Phase 1: Spatial Latin Hypercube Sampling
        if num_samples_target > 0 and n_groups > 0:

            # A. Feature Engineering
            group_features = []
            for g_indices in groups:
                sub = internal_df.iloc[g_indices]
                avg_x = (sub['start_x'].mean() + sub['end_x'].mean()) / 2
                avg_y = (sub['start_y'].mean() + sub['end_y'].mean()) / 2
                l_type = sub['link_type'].iloc[0]
                group_features.append([avg_x, avg_y, l_type])

            data_matrix = np.array(group_features)

            # B. Normalization
            min_vals = data_matrix.min(axis=0)
            max_vals = data_matrix.max(axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1.0
            norm_matrix = (data_matrix - min_vals) / range_vals

            # C. LHS Generation
            sampler = qmc.LatinHypercube(d=3, seed=seed)
            ideal_points = sampler.random(n=num_samples_target)

            # D. KDTree Query
            tree = cKDTree(norm_matrix)
            k_neighbors = min(10, n_groups)
            dists, neighbors_indices = tree.query(ideal_points, k=k_neighbors)

            for i in range(num_samples_target):
                candidates = neighbors_indices[i] if k_neighbors > 1 else [neighbors_indices[i]]

                for candidate_idx in candidates:
                    if candidate_idx >= n_groups: continue

                    if candidate_idx not in selected_indices_set:
                        selected_indices_set.add(candidate_idx)
                        selected_groups.append(groups[candidate_idx])
                        break

                        # 5. Phase 2: Random Fill (Conditional)
        current_count = len(selected_groups)
        missing_count = num_samples_target - current_count

        if missing_count > 0:
            if enable_random_fill:
                logger.info(
                    f"LHS Saturation: {current_count}/{num_samples_target} selected. Filling {missing_count} randomly (random_fill=True).")

                all_indices = set(range(n_groups))
                available_indices = list(all_indices - selected_indices_set)

                rng = np.random.default_rng(seed)
                if len(available_indices) >= missing_count:
                    fill_indices = rng.choice(available_indices, size=missing_count, replace=False)
                    for idx in fill_indices:
                        selected_groups.append(groups[idx])
                        selected_indices_set.add(idx)
                else:
                    logger.warning("Not enough remaining groups to fill target size completely.")
                    for idx in available_indices:
                        selected_groups.append(groups[idx])
            else:
                # If random_fill is False, we accept the smaller sample size
                logger.warning(f"LHS Saturation: {current_count}/{num_samples_target} selected.")
                logger.warning(
                    f"   -> Skipping random fill (random_fill=False). Dataset will be smaller than target rate.")

        # 6. Reconstruct Mask
        sampled_flow_mask = utils.build_mask_from_groups(train_flow_mask, selected_groups)

        # --- LOGGING: Final Results ---
        n_final_links = int(sampled_flow_mask.sum())
        achieved_rate = n_final_links / n_raw_links if n_raw_links > 0 else 0

        logger.info(f"[Sampling Result]")
        logger.info(f"   - Target Groups: {num_samples_target}")
        logger.info(f"   - Final Groups:  {len(selected_groups)}")
        logger.info(f"   - Random Fill Used: {enable_random_fill and missing_count > 0}")
        logger.info(f"   - Final Train Links: {n_final_links} / {n_raw_links} ({achieved_rate * 100:.1f}%)")

        # 7. OD Sampling
        sampled_od_mask = utils.sample_od_simple(od_mask, od_rate, seed)

        return sampled_flow_mask, sampled_od_mask