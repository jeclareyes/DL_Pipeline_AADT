import logging
import os
import json
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
from torch.nn.utils.parametrizations import spectral_norm

logger = logging.getLogger(__name__)


class ODEncoder(nn.Module):
    """Encodes link/OD vectors into latent features."""

    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, 1), feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GraphMatcher(nn.Module):
    """Matcher shared with the CGAME family."""

    def __init__(
        self,
        feature_dim: int,
        num_structures: int,
        lambda_m: float = 0.01,
        lambda_v: float = 0.01,
        reg_strength: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_structures = num_structures
        self.lambda_m = lambda_m
        self.lambda_v = lambda_v
        self.reg_strength = reg_strength

        self.register_buffer("M", torch.randn(feature_dim, num_structures) * 0.1)
        self.register_buffer("V", torch.ones(1, num_structures))
        self.register_buffer("update_count", torch.tensor(0.0))

        self.attention_net = nn.Sequential(nn.Linear(feature_dim, num_structures), nn.Softmax(dim=-1))

    def _ensure_batch_dim(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            return h.unsqueeze(0)
        return h

    @torch.no_grad()
    def update(self, h_x: torch.Tensor, h_y: torch.Tensor):
        h_x = self._ensure_batch_dim(h_x)
        h_y = self._ensure_batch_dim(h_y)

        h_x_norm = F.normalize(h_x, p=2, dim=1)
        h_y_norm = F.normalize(h_y, p=2, dim=1)

        similarity_vector = (h_x_norm * h_y_norm).mean(dim=0)
        target_M = similarity_vector.unsqueeze(1).expand(-1, self.num_structures)

        noise = torch.randn_like(self.M) * 0.01
        regularized_target = target_M + self.reg_strength * noise

        momentum = min(self.lambda_m * (1 + self.update_count.item() * 0.001), 0.1)
        self.M.data = (1 - momentum) * self.M.data + momentum * regularized_target

        h_x_transformed = h_x.unsqueeze(2) * self.M
        h_y_expanded = h_y.unsqueeze(2)

        num = (h_x_transformed * h_y_expanded).sum(dim=1)
        den = torch.norm(h_x_transformed, dim=1) * torch.norm(h_y_expanded, dim=1) + 1e-8
        cosine_sim = (num / den).mean(dim=0)

        target_V = torch.clamp(cosine_sim, min=0.1, max=2.0).unsqueeze(0)
        momentum_v = min(self.lambda_v * (1 + self.update_count.item() * 0.001), 0.1)
        self.V.data = (1 - momentum_v) * self.V.data + momentum_v * target_V

        self.update_count += 1

    def forward(self, h_x: torch.Tensor) -> torch.Tensor:
        h_x = self._ensure_batch_dim(h_x)
        attn_weights = self.attention_net(h_x)

        h_exp = h_x.unsqueeze(2)
        h_struct = h_exp * self.M.unsqueeze(0)
        h_weighted = h_struct * self.V.unsqueeze(0)
        g_x = (h_weighted * attn_weights.unsqueeze(1)).sum(dim=2)
        return g_x


class PositiveLinearBlock(nn.Module):
    """Monotone/Lipschitz-friendly block using positive weights and spectral normalization."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = spectral_norm(nn.Linear(in_dim, out_dim))
        self.activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.softplus(self.linear.weight)
        out = F.linear(x, weight, self.linear.bias)
        return self.activation(out)


class AttributeNet(nn.Module):
    """Learns path-relevant attributes from link, node proxy and path-level features."""

    def __init__(
        self,
        link_in_dim: int,
        node_in_dim: int,
        path_in_dim: int,
        link_attr_dim: int,
        node_attr_dim: int,
        path_attr_dim: int,
    ):
        super().__init__()
        self.link_block = PositiveLinearBlock(link_in_dim, link_attr_dim)
        self.node_block = PositiveLinearBlock(node_in_dim, node_attr_dim)
        self.path_block = PositiveLinearBlock(path_in_dim, path_attr_dim)
        self.total_attr_dim = link_attr_dim + node_attr_dim + path_attr_dim

    def forward(
        self,
        link_features: torch.Tensor,
        node_features: torch.Tensor,
        path_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # link_features: [B, L, F_l], node_features: [B, OD, K, F_n], path_features: [B, OD, K, F_p]
        bsz, num_links, _ = link_features.shape
        link_attrs = self.link_block(link_features.reshape(-1, link_features.shape[-1])).reshape(bsz, num_links, -1)

        node_attrs = self.node_block(node_features.reshape(-1, node_features.shape[-1])).reshape(
            bsz, node_features.shape[1], node_features.shape[2], -1
        )
        path_attrs = self.path_block(path_features.reshape(-1, path_features.shape[-1])).reshape(
            bsz, path_features.shape[1], path_features.shape[2], -1
        )
        return link_attrs, node_attrs, path_attrs


class WeightNet(nn.Module):
    """Maps OD context to non-negative OD-specific attribute weights."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.Softplus(),
        )

    def forward(self, od_features: torch.Tensor) -> torch.Tensor:
        # od_features: [B, OD, F]
        bsz, num_od, feat = od_features.shape
        out = self.network(od_features.reshape(-1, feat)).reshape(bsz, num_od, -1)
        return out


class InverseDemandNet(nn.Module):
    """Approximate inverse demand utility from excess demand and OD features."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, OD, F]
        bsz, num_od, feat = features.shape
        return self.network(features.reshape(-1, feat)).reshape(bsz, num_od)


class PhysicsAwareMirrorDescentAssignment(nn.Module):
    """Path-based implicit layer solved with N-step mirror descent."""

    def __init__(
        self,
        delta_matrix: torch.Tensor,
        route_validity_mask: torch.Tensor,
        od_pair_indices: torch.Tensor,
        t0: torch.Tensor,
        capacity: torch.Tensor,
        lanes: torch.Tensor,
        num_link_groups: int,
        link_group: torch.Tensor,
        vdf_config: DictConfig,
        feature_dim: int,
        physics_cfg: Dict,
    ):
        super().__init__()
        self.register_buffer("delta", delta_matrix.coalesce())
        self.register_buffer("validity_mask", route_validity_mask.bool())
        self.register_buffer("od_pair_indices", od_pair_indices.long())

        self.num_od = int(self.validity_mask.shape[0])
        self.k_paths = int(self.validity_mask.shape[1])
        self.num_routes = int(self.num_od * self.k_paths)
        self.num_links = int(self.delta.size(0))

        route_link_counts = torch.sparse.sum(self.delta, dim=0).to_dense().float()
        self.register_buffer("route_link_counts", torch.clamp(route_link_counts, min=1.0))

        # Gamma route->OD in dense format for optional diagnostics.
        gamma = torch.zeros(self.num_routes, self.num_od, dtype=torch.float32, device=self.validity_mask.device)
        for od in range(self.num_od):
            start = od * self.k_paths
            gamma[start : start + self.k_paths, od] = 1.0
        self.register_buffer("gamma", gamma)

        self.vdf = hydra.utils.instantiate(
            vdf_config,
            t0=t0,
            capacity=capacity,
            lanes=lanes,
            num_link_groups=num_link_groups,
            link_group=link_group,
        )

        attr_cfg = physics_cfg.get("attribute_net", {})
        link_attr_dim = int(attr_cfg.get("link_attr_dim", 4))
        node_attr_dim = int(attr_cfg.get("node_attr_dim", 2))
        path_attr_dim = int(attr_cfg.get("path_attr_dim", 2))

        self.attribute_net = AttributeNet(
            link_in_dim=6,
            node_in_dim=3,
            path_in_dim=4,
            link_attr_dim=link_attr_dim,
            node_attr_dim=node_attr_dim,
            path_attr_dim=path_attr_dim,
        )

        weight_hidden = int(physics_cfg.get("weight_hidden_dim", max(feature_dim // 2, 8)))
        self.weight_net = WeightNet(in_dim=3, hidden_dim=weight_hidden, out_dim=self.attribute_net.total_attr_dim)

        # Link-level static features used by AttributeNet.
        eps = 1e-6
        t0_n = t0 / (t0.mean() + eps)
        cap_n = capacity / (capacity.mean() + eps)
        lane_n = lanes.float() / (lanes.float().mean() + eps)
        speed_n = t0 / (torch.clamp(t0, min=eps).mean() + eps)  # fallback proxy if speed is unavailable
        group_n = link_group.float() / max(float(num_link_groups - 1), 1.0)
        ones = torch.ones_like(t0_n)
        static_link_features = torch.stack([ones, t0_n, cap_n, lane_n, speed_n, group_n], dim=1)
        self.register_buffer("static_link_features", static_link_features)

        od_idx = od_pair_indices.float()
        od_norm = od_idx / (torch.clamp(od_idx.max(), min=1.0))
        self.register_buffer("od_norm_features", od_norm)

        # Cached warm-start state.
        self.register_buffer("cached_sigma", torch.zeros(1, self.num_routes))

    def _uniform_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        valid = self.validity_mask.float().to(device)
        denom = torch.clamp(valid.sum(dim=1, keepdim=True), min=1.0)
        sigma = (valid / denom).unsqueeze(0).expand(batch_size, -1, -1)
        return sigma.reshape(batch_size, -1)

    def _normalize_sigma(self, sigma3: torch.Tensor) -> torch.Tensor:
        valid = self.validity_mask.unsqueeze(0).expand(sigma3.shape[0], -1, -1)
        sigma3 = torch.where(valid, sigma3, torch.zeros_like(sigma3))
        denom = sigma3.sum(dim=2, keepdim=True)
        fallback = self._uniform_sigma(sigma3.shape[0], sigma3.device).reshape(sigma3.shape)
        normalized = torch.where(denom > 1e-12, sigma3 / (denom + 1e-12), fallback)
        return normalized

    def _build_od_features(self, q: torch.Tensor) -> torch.Tensor:
        # q: [B, OD]
        od_norm = self.od_norm_features.unsqueeze(0).expand(q.shape[0], -1, -1)
        q_norm = q / (q.mean(dim=1, keepdim=True) + 1e-6)
        return torch.cat([q_norm.unsqueeze(-1), od_norm], dim=-1)

    def _compute_perceived_cost(
        self,
        q: torch.Tensor,
        h: torch.Tensor,
        link_flows: torch.Tensor,
        link_times: torch.Tensor,
    ) -> torch.Tensor:
        bsz = q.shape[0]
        route_times = torch.sparse.mm(self.delta.t(), link_times.t()).t()

        # Build features for nets.
        link_features = self.static_link_features.unsqueeze(0).expand(bsz, -1, -1).clone()
        link_features[:, :, 0] = link_flows / (link_flows.mean(dim=1, keepdim=True) + 1e-6)

        h3 = h.view(bsz, self.num_od, self.k_paths)
        q3 = q.unsqueeze(-1).expand(-1, -1, self.k_paths)
        route_lengths = self.route_link_counts.view(self.num_od, self.k_paths).unsqueeze(0).expand(bsz, -1, -1)

        node_features = torch.stack(
            [
                h3 / (h3.mean(dim=(1, 2), keepdim=True) + 1e-6),
                q3 / (q3.mean(dim=(1, 2), keepdim=True) + 1e-6),
                route_lengths / (route_lengths.mean(dim=(1, 2), keepdim=True) + 1e-6),
            ],
            dim=-1,
        )

        route_times3 = route_times.view(bsz, self.num_od, self.k_paths)
        path_features = torch.stack(
            [
                route_times3 / (route_times3.mean(dim=(1, 2), keepdim=True) + 1e-6),
                h3 / (h3.mean(dim=(1, 2), keepdim=True) + 1e-6),
                q3 / (q3.mean(dim=(1, 2), keepdim=True) + 1e-6),
                route_lengths / (route_lengths.mean(dim=(1, 2), keepdim=True) + 1e-6),
            ],
            dim=-1,
        )

        link_attrs, node_attrs, path_attrs = self.attribute_net(link_features, node_features, path_features)

        # Aggregate link attrs to route attrs via delta^T.
        route_link_attrs = []
        for d in range(link_attrs.shape[-1]):
            route_attr_d = torch.sparse.mm(self.delta.t(), link_attrs[:, :, d].t()).t()
            route_link_attrs.append(route_attr_d.unsqueeze(-1))
        route_link_attrs = torch.cat(route_link_attrs, dim=-1).view(bsz, self.num_od, self.k_paths, -1)

        route_attr_all = torch.cat([route_link_attrs, node_attrs, path_attrs], dim=-1)

        od_features = self._build_od_features(q)
        od_weights = self.weight_net(od_features)
        route_weights = od_weights.unsqueeze(2).expand(-1, -1, self.k_paths, -1)

        weighted_attr_cost = (route_weights * route_attr_all).sum(dim=-1)
        perceived = route_times3 + weighted_attr_cost
        return perceived.reshape(bsz, self.num_routes)

    def forward(
        self,
        od_demand: torch.Tensor,
        eta: float,
        max_iters: int,
        min_iters: int,
        tol_rel_flow: float,
        use_early_stop: bool,
        warm_start: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # od_demand: [B, OD]
        bsz = od_demand.shape[0]
        device = od_demand.device

        if warm_start and self.cached_sigma.shape[1] == self.num_routes and self.cached_sigma.shape[0] == 1:
            sigma = self.cached_sigma.expand(bsz, -1).to(device)
        else:
            sigma = self._uniform_sigma(bsz, device)

        sigma = self._normalize_sigma(sigma.view(bsz, self.num_od, self.k_paths)).reshape(bsz, self.num_routes)

        q_flat = od_demand.unsqueeze(-1).expand(-1, -1, self.k_paths).reshape(bsz, self.num_routes)
        prev_flow = None
        last_rel_mean = torch.tensor(0.0, device=device)
        converged_flag = torch.zeros(bsz, dtype=torch.bool, device=device)
        used_iters = max_iters

        for n in range(1, max_iters + 1):
            route_flows = sigma * q_flat
            link_flows = torch.sparse.mm(self.delta, route_flows.t()).t()
            link_times = self.vdf(link_flows)
            perceived = self._compute_perceived_cost(od_demand, route_flows, link_flows, link_times)

            c3 = perceived.view(bsz, self.num_od, self.k_paths)
            valid = self.validity_mask.unsqueeze(0).expand(bsz, -1, -1)
            c3_masked = torch.where(valid, c3, torch.full_like(c3, float("inf")))

            c_min = c3_masked.min(dim=2, keepdim=True).values
            c_shift = torch.where(valid, c3 - c_min, torch.zeros_like(c3))
            exp_term = torch.exp(torch.clamp(-eta * c_shift, min=-30.0, max=30.0))

            sigma3 = sigma.view(bsz, self.num_od, self.k_paths)
            sigma_next3 = sigma3 * exp_term
            sigma_next3 = self._normalize_sigma(sigma_next3)
            sigma_next = sigma_next3.reshape(bsz, self.num_routes)

            if prev_flow is not None:
                rel = torch.norm(link_flows - prev_flow, p=2, dim=1) / (torch.norm(prev_flow, p=2, dim=1) + 1e-8)
                last_rel_mean = rel.mean()
                converged = rel < tol_rel_flow
                converged_flag = converged_flag | converged
                if use_early_stop and n >= min_iters and bool(torch.all(converged)):
                    sigma = sigma_next
                    used_iters = n
                    break

            sigma = sigma_next
            prev_flow = link_flows
            used_iters = n

        route_flows = sigma * q_flat
        link_flows = torch.sparse.mm(self.delta, route_flows.t()).t()
        link_times = self.vdf(link_flows)
        route_costs = self._compute_perceived_cost(od_demand, route_flows, link_flows, link_times)

        self.cached_sigma = sigma.detach().mean(dim=0, keepdim=True)

        info = {
            "iterations": torch.tensor(float(used_iters), device=device),
            "converged_ratio": converged_flag.float().mean(),
            "final_gap": last_rel_mean,
        }
        return link_flows, route_flows, route_costs, info


class CyclicODModel(nn.Module):
    def __init__(
        self,
        num_links: int,
        num_od_pairs: int,
        architecture: DictConfig,
        delta_matrix: torch.Tensor,
        route_validity_mask: torch.Tensor,
        od_pair_indices: torch.Tensor,
        t0: torch.Tensor,
        capacity: torch.Tensor,
        lanes: torch.Tensor,
        num_link_groups: int,
        link_group: torch.Tensor,
        vdf_config: DictConfig,
        **kwargs,
    ):
        super().__init__()
        logger.info("INIT: CGAME_PhysicsMirrorDescent")

        arch = architecture
        feature_dim = int(arch.feature_dim)
        num_structures = int(arch.num_structures)
        h_enc = int(arch.hidden_dim_from_link)
        h_dec = int(arch.hidden_dim_to_od)
        h_bwd_enc = int(arch.get("hidden_dim_from_od", h_dec))
        h_bwd_dec = int(arch.get("hidden_dim_to_link", h_enc))
        dropout = float(arch.get("dropout", 0.1))

        self.num_links = int(num_links)
        self.num_od_pairs = int(num_od_pairs)

        self.register_buffer("link_scale", torch.tensor(float(kwargs.get("link_scale", 1.0)), dtype=torch.float32))
        self.register_buffer("od_scale", torch.tensor(float(kwargs.get("od_scale", 1.0)), dtype=torch.float32))

        q_max_cfg = kwargs.get("q_max", None)
        if q_max_cfg is None:
            initial_mean = float(kwargs.get("initial_mean", 1.0))
            q_max_cfg = max(initial_mean * 5.0, 1.0)
        self.register_buffer("q_max", torch.tensor(float(q_max_cfg), dtype=torch.float32))

        self.f_encoder = ODEncoder(num_links, h_enc, feature_dim, dropout)
        self.matcher = GraphMatcher(feature_dim, num_structures)

        # Demand head (flow->OD) plus inverse demand refinement.
        self.demand_head = nn.Sequential(
            nn.Linear(feature_dim, h_dec),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(h_dec, num_od_pairs),
            nn.Softplus(),
        )

        inv_hidden = int(arch.get("inverse_demand_hidden_dim", max(feature_dim // 2, 8)))
        self.inverse_demand_net = InverseDemandNet(in_dim=3, hidden_dim=inv_hidden)

        # Data-driven fallback branch for warmup phases.
        self.b_encoder = ODEncoder(num_od_pairs, h_bwd_enc, feature_dim, dropout)
        self.b_decoder = nn.Sequential(
            nn.Linear(feature_dim, h_bwd_dec),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(h_bwd_dec, num_links),
            nn.Softplus(),
        )

        hybrid_cfg = kwargs.get("hybrid_assignment", {})
        if isinstance(hybrid_cfg, DictConfig):
            hybrid_cfg = dict(hybrid_cfg)
        self.hybrid_use_physics = bool(hybrid_cfg.get("use_physics", True))
        self.hybrid_warmup_epochs = int(hybrid_cfg.get("warmup_epochs", 30))

        md_cfg = kwargs.get("mirror_descent", {})
        if isinstance(md_cfg, DictConfig):
            md_cfg = dict(md_cfg)
        self.md_cfg = {
            "eta": float(md_cfg.get("eta", 1.0)),
            "max_iters": int(md_cfg.get("max_iters", 30)),
            "min_iters": int(md_cfg.get("min_iters", 5)),
            "tol_rel_flow": float(md_cfg.get("tol_rel_flow", 1e-4)),
            "use_early_stop": bool(md_cfg.get("use_early_stop", True)),
            "warm_start": bool(md_cfg.get("warm_start", True)),
            "adaptive_n": bool(md_cfg.get("adaptive_n", False)),
            "adaptive_n_max": int(md_cfg.get("adaptive_n_max", 80)),
            "adaptive_n_ramp_epochs": int(md_cfg.get("adaptive_n_ramp_epochs", 100)),
        }

        physics_cfg = kwargs.get("physics", {})
        if isinstance(physics_cfg, DictConfig):
            physics_cfg = dict(physics_cfg)

        self.assignment = PhysicsAwareMirrorDescentAssignment(
            delta_matrix=delta_matrix,
            route_validity_mask=route_validity_mask,
            od_pair_indices=od_pair_indices,
            t0=t0,
            capacity=capacity,
            lanes=lanes,
            num_link_groups=num_link_groups,
            link_group=link_group,
            vdf_config=vdf_config,
            feature_dim=feature_dim,
            physics_cfg=physics_cfg,
        )

        loss_cfg = kwargs.get("loss", {})
        if isinstance(loss_cfg, DictConfig):
            loss_cfg = dict(loss_cfg)
        else:
            loss_cfg = dict(loss_cfg) if isinstance(loss_cfg, dict) else {}
        loss_cfg.pop("_target_", None)

        link_scale_cfg = loss_cfg.pop("link_scale", float(self.link_scale.item()))
        od_scale_cfg = loss_cfg.pop("od_scale", float(self.od_scale.item()))
        self.link_scale.fill_(float(link_scale_cfg))
        self.od_scale.fill_(float(od_scale_cfg))

        self.loss_fn = Loss(
            link_scale=float(self.link_scale.item()),
            od_scale=float(self.od_scale.item()),
            **loss_cfg,
        )

    def _ensure_batch(self, x: torch.Tensor) -> torch.Tensor:
        if x is None:
            return x
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x

    def _adaptive_iters(self, current_epoch: Optional[int]) -> int:
        if not self.md_cfg["adaptive_n"] or current_epoch is None:
            return self.md_cfg["max_iters"]
        ramp = max(self.md_cfg["adaptive_n_ramp_epochs"], 1)
        frac = min(max(float(current_epoch) / float(ramp), 0.0), 1.0)
        return int(round(self.md_cfg["max_iters"] + frac * (self.md_cfg["adaptive_n_max"] - self.md_cfg["max_iters"])))

    def _refine_od_with_inverse_demand(self, raw_od_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = raw_od_hat.shape[0]
        od_idx = self.assignment.od_pair_indices.float().to(raw_od_hat.device)
        od_norm = od_idx / (torch.clamp(od_idx.max(), min=1.0))
        od_norm = od_norm.unsqueeze(0).expand(bsz, -1, -1)

        raw_physical = raw_od_hat * self.od_scale
        q_max = torch.clamp(self.q_max, min=1.0)
        excess = torch.clamp(q_max - raw_physical, min=0.0) / q_max
        inv_features = torch.cat([excess.unsqueeze(-1), od_norm], dim=-1)

        utility = self.inverse_demand_net(inv_features)
        adjusted = raw_physical * (1.0 + 0.1 * torch.tanh(utility))
        od_hat = torch.clamp(adjusted, min=0.0, max=q_max)
        return od_hat, utility

    def forward(
        self,
        observed_flows: torch.Tensor,
        flow_mask: Optional[torch.Tensor] = None,
        true_od_demand: Optional[torch.Tensor] = None,
        od_mask: Optional[torch.Tensor] = None,
        warmup: bool = False,
        current_epoch: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        observed_flows = self._ensure_batch(observed_flows)
        flow_mask = self._ensure_batch(flow_mask) if flow_mask is not None else torch.ones_like(observed_flows)
        true_od_demand = self._ensure_batch(true_od_demand) if true_od_demand is not None else None
        od_mask = self._ensure_batch(od_mask) if od_mask is not None else None

        x_in = (observed_flows / self.link_scale) * flow_mask
        hx = self.f_encoder(x_in)
        gx = self.matcher(hx)

        raw_od_hat = self.demand_head(gx)
        od_hat, od_utility = self._refine_od_with_inverse_demand(raw_od_hat)

        use_teacher = self.training and (true_od_demand is not None)
        if use_teacher:
            if od_mask is not None:
                od_for_assignment = (true_od_demand * od_mask) + (od_hat * (1.0 - od_mask))
            else:
                od_for_assignment = true_od_demand
        else:
            od_for_assignment = od_hat

        epoch_in_warmup = current_epoch is not None and current_epoch < self.hybrid_warmup_epochs
        use_physics = self.hybrid_use_physics and not (warmup or epoch_in_warmup)

        if use_physics:
            iters = self._adaptive_iters(current_epoch)
            link_flows_phys, route_flows, route_costs, conv_info = self.assignment(
                od_demand=od_for_assignment,
                eta=self.md_cfg["eta"],
                max_iters=iters,
                min_iters=self.md_cfg["min_iters"],
                tol_rel_flow=self.md_cfg["tol_rel_flow"],
                use_early_stop=self.md_cfg["use_early_stop"],
                warm_start=self.md_cfg["warm_start"] and self.training,
            )
            flow_hat = link_flows_phys
            hy = None
            gy = None
            mode = "physics"
        else:
            od_norm = od_for_assignment / self.od_scale
            hy = self.b_encoder(od_norm)
            gy = self.matcher(hy)
            raw_flow_hat = self.b_decoder(gy)
            flow_hat = raw_flow_hat * self.link_scale
            route_flows = None
            route_costs = None
            conv_info = {
                "iterations": torch.tensor(0.0, device=flow_hat.device),
                "converged_ratio": torch.tensor(0.0, device=flow_hat.device),
            }
            mode = "data_driven"

        if self.training and hy is not None:
            self.matcher.update(hx.detach(), hy.detach())

        output = {
            "estimated_demand": od_hat,
            "reconstructed_flows": flow_hat,
            "hx": hx,
            "gx": gx,
            "hy": hy,
            "gy": gy,
            "od_utility": od_utility,
            "route_flows": route_flows,
            "route_costs": route_costs,
            "convergence_info": {
                "iterations": conv_info["iterations"],
                "converged_ratio": conv_info["converged_ratio"],
                "final_gap": conv_info.get("final_gap", torch.tensor(0.0, device=flow_hat.device)),
                "mode": mode,
            },
        }

        if true_od_demand is not None:
            output["loss"] = self.loss_fn(
                predicted_flows=flow_hat,
                true_flows=observed_flows,
                flow_mask=flow_mask,
                predicted_od=od_hat,
                true_od=true_od_demand,
                od_mask=od_mask,
                route_costs=route_costs,
                route_flows=route_flows,
            )

        return output

    def get_evaluation_artifacts(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert isinstance(outputs, dict), "outputs must be a dict"
        assert "reconstructed_flows" in outputs, "Missing key 'reconstructed_flows'"
        assert "estimated_demand" in outputs, "Missing key 'estimated_demand'"
        return {
            "pred_flows": outputs["reconstructed_flows"].detach().cpu(),
            "pred_od": outputs["estimated_demand"].detach().cpu(),
            "route_flows": outputs.get("route_flows"),
            "route_costs": outputs.get("route_costs"),
            "hx": outputs.get("hx"),
            "gx": outputs.get("gx"),
            "hy": outputs.get("hy"),
            "gy": outputs.get("gy"),
            "convergence_info": outputs.get("convergence_info"),
        }


class Loss(nn.Module):
    def __init__(
        self,
        link_scale: float = 1.0,
        od_scale: float = 1.0,
        w_flow: float = 1.0,
        w_od: float = 1.0,
        w_gap: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.register_buffer("link_scale", torch.tensor(float(link_scale)))
        self.register_buffer("od_scale", torch.tensor(float(od_scale)))
        self.w_flow = float(w_flow)
        self.w_od = float(w_od)
        self.w_gap = float(w_gap)
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        pred_flow = kwargs.get("predicted_flows")
        true_flow = kwargs.get("true_flows")
        flow_mask = kwargs.get("flow_mask")

        pred_od = kwargs.get("predicted_od")
        true_od = kwargs.get("true_od")
        od_mask = kwargs.get("od_mask")

        if pred_flow is None or true_flow is None:
            z = torch.tensor(0.0, device=self.link_scale.device, requires_grad=True)
            return {"total_loss": z, "l_flow": z.detach(), "l_od": z.detach(), "l_gap": z.detach()}

        if flow_mask is None:
            flow_mask = torch.ones_like(true_flow)

        scaled_pred_f = pred_flow / self.link_scale
        scaled_true_f = true_flow / self.link_scale
        l_flow = (self.mse(scaled_pred_f, scaled_true_f) * flow_mask).sum() / (flow_mask.sum() + 1e-6)

        l_od = torch.tensor(0.0, device=pred_flow.device)
        if pred_od is not None and true_od is not None and od_mask is not None and od_mask.sum() > 0:
            scaled_pred_od = pred_od / self.od_scale
            scaled_true_od = true_od / self.od_scale
            l_od = (self.mse(scaled_pred_od, scaled_true_od) * od_mask).sum() / (od_mask.sum() + 1e-6)

        l_gap = torch.tensor(0.0, device=pred_flow.device)
        route_costs = kwargs.get("route_costs")
        route_flows = kwargs.get("route_flows")
        if route_costs is not None and route_flows is not None:
            # Proxy: used-route cost dispersion.
            used = route_flows > 1e-6
            if used.any():
                costs_used = torch.where(used, route_costs, torch.zeros_like(route_costs))
                mean_used = costs_used.sum(dim=1, keepdim=True) / (used.float().sum(dim=1, keepdim=True) + 1e-6)
                l_gap = ((costs_used - mean_used).abs() * used.float()).sum() / (used.float().sum() + 1e-6)

        total_loss = (self.w_flow * l_flow) + (self.w_od * l_od) + (self.w_gap * l_gap)
        return {"total_loss": total_loss, "l_flow": l_flow, "l_od": l_od, "l_gap": l_gap}


class TrainingDiagnostician:
    def __init__(self, history_window: int = 100):
        self.window = int(history_window)
        self.full_history = {
            "r2_flow": [],
            "mae_flow": [],
            "l_flow": [],
            "l_od": [],
            "l_gap": [],
            "total_loss": [],
            "iterations": [],
            "converged_ratio": [],
            "final_gap": [],
            "mode_is_physics": [],
            "mode": [],
            "max_grad": [],
            "grad_norms": {},
            "alerts": [],
        }
        self.window_history = {"r2_flow": [], "mae_flow": [], "max_grad": []}

        # Conservative defaults for mixed data-driven/physics phases.
        self.grad_high_threshold = 1e3
        self.grad_low_threshold = 1e-7
        self.spike_ratio_threshold = 8.0

    def update(self, outputs: Dict, targets: Dict, model=None, **kwargs):
        with torch.no_grad():
            pred_flow = outputs.get("reconstructed_flows")
            true_flow = targets.get("flows")
            mask_flow = targets.get("flow_mask")
            if mask_flow is None:
                mask_flow = targets.get("mask")

            if pred_flow is None or true_flow is None:
                return
            if mask_flow is None:
                mask_flow = torch.ones_like(true_flow)

            if pred_flow.dim() == 2 and pred_flow.shape[0] == 1:
                pred_flow = pred_flow.squeeze(0)
            if true_flow.dim() == 2 and true_flow.shape[0] == 1:
                true_flow = true_flow.squeeze(0)
            if mask_flow.dim() == 2 and mask_flow.shape[0] == 1:
                mask_flow = mask_flow.squeeze(0)

            mask_bool = mask_flow > 0
            if mask_bool.sum() <= 0:
                return

            y_pred = pred_flow[mask_bool].detach().cpu().numpy()
            y_true = true_flow[mask_bool].detach().cpu().numpy()

            mae = float(np.mean(np.abs(y_true - y_pred)))
            if len(y_true) > 1 and np.var(y_true) > 0:
                ss_res = float(np.sum((y_true - y_pred) ** 2))
                ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
                r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
            else:
                r2 = 0.0

            self.full_history["r2_flow"].append(r2)
            self.full_history["mae_flow"].append(mae)
            self._push_window("r2_flow", r2)
            self._push_window("mae_flow", mae)

            loss_dict = kwargs.get("loss_dict") or outputs.get("loss") or {}
            if isinstance(loss_dict, dict):
                self._append_loss("total_loss", loss_dict.get("total_loss"))
                self._append_loss("l_flow", loss_dict.get("l_flow"))
                self._append_loss("l_od", loss_dict.get("l_od"))
                self._append_loss("l_gap", loss_dict.get("l_gap"))

            conv_info = outputs.get("convergence_info", {}) or {}
            self._append_scalar("iterations", conv_info.get("iterations"), default=0.0)
            self._append_scalar("converged_ratio", conv_info.get("converged_ratio"), default=0.0)
            self._append_scalar("final_gap", conv_info.get("final_gap"), default=0.0)
            mode = str(conv_info.get("mode", "data_driven"))
            self.full_history["mode_is_physics"].append(1.0 if mode == "physics" else 0.0)
            self.full_history["mode"].append(mode)
            self._record_physics_alerts(mode=mode)

    def capture_gradient_history(self, model: nn.Module, epoch: int):
        if "grad_norms" not in self.full_history:
            self.full_history["grad_norms"] = {}

        key_modules = {
            "Fwd Enc": getattr(model, "f_encoder", None),
            "Matcher": getattr(model, "matcher", None),
            "Demand Head": getattr(model, "demand_head", None),
            "Assignment": getattr(model, "assignment", None),
            "Inverse Demand": getattr(model, "inverse_demand_net", None),
        }

        max_grad = 0.0
        module_norms = {}
        for name, module in key_modules.items():
            if module is None:
                continue
            total_norm = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            module_norms[name] = total_norm
            self.full_history["grad_norms"].setdefault(name, []).append(total_norm)
            max_grad = max(max_grad, total_norm)

        self.full_history["max_grad"].append(max_grad)
        self._push_window("max_grad", max_grad)
        self._record_gradient_alerts(epoch=epoch, max_grad=max_grad)
        self._record_module_gradient_alerts(epoch=epoch, module_norms=module_norms)

    def get_report(self) -> str:
        if not self.window_history["r2_flow"]:
            return "Init..."
        avg_r2 = np.mean(self.window_history["r2_flow"])
        avg_mae = np.mean(self.window_history["mae_flow"])
        avg_max_grad = np.mean(self.window_history["max_grad"]) if self.window_history["max_grad"] else 0.0
        return f"R2: {avg_r2:.3f} | MAE: {avg_mae:.1f} | MaxGrad: {avg_max_grad:.2e}"

    def _to_float(self, value, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        if torch.is_tensor(value):
            if value.numel() == 0:
                return float(default)
            return float(value.detach().mean().cpu().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _append_scalar(self, key: str, value, default: float = 0.0):
        self.full_history[key].append(self._to_float(value, default=default))

    def _append_loss(self, key: str, value):
        self._append_scalar(key, value, default=0.0)

    def _record_gradient_alerts(self, epoch: int, max_grad: float):
        if max_grad > self.grad_high_threshold:
            self.full_history["alerts"].append((epoch, "high_grad", max_grad))
            return
        if max_grad < self.grad_low_threshold and epoch > 2:
            self.full_history["alerts"].append((epoch, "low_grad", max_grad))
            return
        if len(self.full_history["max_grad"]) >= 2:
            prev = max(self.full_history["max_grad"][-2], 1e-12)
            ratio = max_grad / prev
            if ratio > self.spike_ratio_threshold:
                self.full_history["alerts"].append((epoch, "spike_grad", max_grad))

    def _record_module_gradient_alerts(self, epoch: int, module_norms: Dict[str, float]):
        if not module_norms:
            return

        if self.full_history["mode_is_physics"] and self.full_history["mode_is_physics"][-1] > 0.5:
            assignment_norm = module_norms.get("Assignment")
            inverse_norm = module_norms.get("Inverse Demand")
            if assignment_norm is not None and assignment_norm < self.grad_low_threshold and epoch > 2:
                self.full_history["alerts"].append((epoch, "assignment_low_grad", max(assignment_norm, 1e-12)))
            if inverse_norm is not None and inverse_norm < self.grad_low_threshold and epoch > 2:
                self.full_history["alerts"].append((epoch, "inverse_demand_low_grad", max(inverse_norm, 1e-12)))

        positive = [v for v in module_norms.values() if v > 0]
        if len(positive) >= 2:
            median_val = float(np.median(positive))
            dominant = max(positive)
            if median_val > 0 and dominant / median_val > 20.0:
                self.full_history["alerts"].append((epoch, "module_grad_imbalance", dominant))

    def _record_physics_alerts(self, mode: str):
        epoch = len(self.full_history["r2_flow"]) - 1
        if epoch < 0:
            return

        if mode == "physics":
            final_gap = self.full_history["final_gap"][-1] if self.full_history["final_gap"] else 0.0
            conv_ratio = self.full_history["converged_ratio"][-1] if self.full_history["converged_ratio"] else 0.0
            if final_gap > 0.2:
                self.full_history["alerts"].append((epoch, "high_final_gap", final_gap))
            if conv_ratio < 0.1 and epoch > 2:
                self.full_history["alerts"].append((epoch, "low_convergence_ratio", max(conv_ratio, 1e-12)))

        total = self.full_history["total_loss"][-1] if self.full_history["total_loss"] else 0.0
        l_flow = self.full_history["l_flow"][-1] if self.full_history["l_flow"] else 0.0
        l_od = self.full_history["l_od"][-1] if self.full_history["l_od"] else 0.0
        l_gap = self.full_history["l_gap"][-1] if self.full_history["l_gap"] else 0.0
        comp_sum = max(l_flow + l_od + l_gap, 1e-12)
        dominant_name, dominant_val = max(
            (("l_flow", l_flow), ("l_od", l_od), ("l_gap", l_gap)),
            key=lambda x: x[1],
        )
        if total > 0 and dominant_val / comp_sum > 0.95 and epoch > 5:
            self.full_history["alerts"].append((epoch, f"dominant_{dominant_name}", dominant_val))

    def _push_window(self, key: str, value: float):
        self.window_history[key].append(value)
        if len(self.window_history[key]) > self.window:
            self.window_history[key].pop(0)

    def finalize_and_plot(self, filename_prefix: str = "final_report"):
        plt.switch_backend("Agg")
        base_dir = os.path.dirname(filename_prefix)
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
        self.plot_evolution(filename_prefix.replace(".png", "_evolution.png"))
        self.plot_physics(filename_prefix.replace(".png", "_physics.png"))
        self.plot_gradient_health(filename_prefix.replace(".png", "_gradients.png"))

    def plot_evolution(self, filename: str):
        r2 = self.full_history["r2_flow"]
        mae = self.full_history["mae_flow"]
        if not r2:
            return

        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("R2 Score", color="tab:blue")
        ax1.plot(r2, color="tab:blue", label="R2 Flow", alpha=0.7)
        ax1.set_ylim(-1, 1)

        ax2 = ax1.twinx()
        ax2.set_ylabel("MAE Flow", color="tab:orange")
        ax2.plot(mae, color="tab:orange", label="MAE Flow", alpha=0.7)

        plt.title("Training Evolution")
        plt.savefig(filename)
        plt.close()

    def plot_physics(self, filename: str):
        loss_flow = self.full_history["l_flow"]
        loss_od = self.full_history["l_od"]
        loss_gap = self.full_history["l_gap"]
        iters = self.full_history["iterations"]
        conv_ratio = self.full_history["converged_ratio"]
        final_gap = self.full_history["final_gap"]

        if not loss_flow and not iters:
            return

        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

        if loss_flow:
            ax1.plot(loss_flow, label="l_flow", alpha=0.85)
        if loss_od:
            ax1.plot(loss_od, label="l_od", alpha=0.85)
        if loss_gap:
            ax1.plot(loss_gap, label="l_gap", alpha=0.85)
        ax1.set_ylabel("Loss Terms")
        ax1.legend(loc="upper right")
        ax1.grid(alpha=0.2)

        if iters:
            ax2.plot(iters, label="iterations", color="tab:blue", alpha=0.85)
        if conv_ratio:
            ax2.plot(conv_ratio, label="converged_ratio", color="tab:green", alpha=0.85)
        if final_gap:
            ax2.plot(final_gap, label="final_gap", color="tab:red", alpha=0.85)
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Physics Convergence")
        ax2.grid(alpha=0.2)
        ax2.legend(loc="upper right")

        plt.suptitle("PhysicsMirrorDescent Diagnostics")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_gradient_health(self, filename: str):
        grads = self.full_history.get("grad_norms", {})
        if not grads:
            return
        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(12, 6))
        for name, values in grads.items():
            if values:
                plt.plot(values, label=name)
        max_grad = self.full_history.get("max_grad", [])
        if max_grad:
            plt.plot(max_grad, label="Max Grad", color="black", linewidth=1.5, alpha=0.75)
        for epoch, kind, value in self.full_history.get("alerts", []):
            color = "tab:red" if kind in ("high_grad", "spike_grad") else "tab:orange"
            plt.scatter(epoch, max(value, 1e-12), color=color, s=16, alpha=0.8)
        plt.yscale("log")
        plt.legend()
        plt.title("Gradient Health")
        plt.savefig(filename)
        plt.close()

    def save_summary(self, filename: str):
        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        alerts = self.full_history.get("alerts", [])
        alert_counts = {}
        for _, kind, _ in alerts:
            alert_counts[kind] = alert_counts.get(kind, 0) + 1

        summary = {
            "steps": len(self.full_history.get("r2_flow", [])),
            "metrics": {
                "r2_flow_mean": float(np.mean(self.full_history.get("r2_flow", [0.0]))),
                "mae_flow_mean": float(np.mean(self.full_history.get("mae_flow", [0.0]))),
                "max_grad_mean": float(np.mean(self.full_history.get("max_grad", [0.0]))),
                "iterations_mean": float(np.mean(self.full_history.get("iterations", [0.0]))),
                "final_gap_mean": float(np.mean(self.full_history.get("final_gap", [0.0]))),
                "physics_mode_ratio": float(np.mean(self.full_history.get("mode_is_physics", [0.0]))),
            },
            "alerts": {
                "total": len(alerts),
                "counts": alert_counts,
                "examples": [
                    {"epoch": int(epoch), "type": kind, "value": float(value)}
                    for epoch, kind, value in alerts[-10:]
                ],
            },
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
