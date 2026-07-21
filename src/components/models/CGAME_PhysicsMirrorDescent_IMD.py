from typing import Dict, Optional, Tuple

import torch
from omegaconf import DictConfig

from .CGAME_PhysicsMirrorDescent import (
    CyclicODModel as BaseCyclicODModel,
    Loss,
    PhysicsAwareMirrorDescentAssignment,
    TrainingDiagnostician,
)


class IMDMirrorDescentFunction(torch.autograd.Function):
    """
    Inexact Implicit Differentiation for the Mirror Descent assignment fixed point.

    Forward: solves the fixed-point iterations without storing the full unrolled graph.
    Backward: approximates the implicit VJP using damped Richardson iterations.
    """

    @staticmethod
    def forward(ctx, assignment_module, sigma_init, od_demand, q_flat, *module_params):
        with torch.no_grad():
            sigma_current = sigma_init
            prev_flow = None
            last_rel_mean = torch.tensor(0.0, device=sigma_init.device)
            converged_flag = torch.zeros(sigma_init.shape[0], dtype=torch.bool, device=sigma_init.device)
            used_iters = assignment_module.current_forward_iters

            for n in range(1, assignment_module.current_forward_iters + 1):
                link_flows = assignment_module._link_flows_from_sigma(sigma_current, q_flat)
                sigma_next = assignment_module._fixed_point_step(sigma_current, od_demand, q_flat)

                if prev_flow is not None:
                    rel = torch.norm(link_flows - prev_flow, p=2, dim=1) / (torch.norm(prev_flow, p=2, dim=1) + 1e-8)
                    last_rel_mean = rel.mean()
                    converged = rel < assignment_module.current_tol_rel_flow
                    converged_flag = converged_flag | converged

                    if (
                        assignment_module.current_use_early_stop
                        and n >= assignment_module.current_min_iters
                        and bool(torch.all(converged))
                    ):
                        sigma_current = sigma_next
                        used_iters = n
                        break

                sigma_current = sigma_next
                prev_flow = link_flows
                used_iters = n

        sigma_star = sigma_current

        assignment_module._imd_last_info = {
            "iterations": float(used_iters),
            "converged_ratio": float(converged_flag.float().mean().item()),
            "final_gap": float(last_rel_mean.item()),
            "implicit_grad": True,
        }

        ctx.assignment_module = assignment_module
        ctx.save_for_backward(sigma_star, od_demand, q_flat, *module_params)
        return sigma_star

    @staticmethod
    def backward(ctx, grad_output):
        sigma_star, od_demand, q_flat = ctx.saved_tensors[:3]
        module_params = ctx.saved_tensors[3:]
        assignment_module = ctx.assignment_module

        with torch.enable_grad():
            sigma_star = sigma_star.detach().requires_grad_(True)
            od_demand = od_demand.detach().requires_grad_(True)
            q_flat = q_flat.detach().requires_grad_(True)

            sigma_next = assignment_module._fixed_point_step(sigma_star, od_demand, q_flat)

            z = grad_output.clone()
            damping = assignment_module.damping_factor

            for _ in range(assignment_module.backward_iters):
                vjp_sigma = torch.autograd.grad(
                    sigma_next,
                    sigma_star,
                    grad_outputs=z,
                    retain_graph=True,
                    allow_unused=False,
                )[0]
                z = (1.0 - damping) * z + damping * (grad_output + vjp_sigma)

            inputs_and_params = (od_demand, q_flat) + module_params
            grads = torch.autograd.grad(
                sigma_next,
                inputs_and_params,
                grad_outputs=z,
                allow_unused=True,
            )

        grad_od_demand = grads[0]
        grad_q_flat = grads[1]
        grad_module_params = grads[2:]

        return (None, None, grad_od_demand, grad_q_flat) + grad_module_params


class PhysicsAwareMirrorDescentAssignmentIMD(PhysicsAwareMirrorDescentAssignment):
    def __init__(self, *args, imd_cfg: Optional[Dict] = None, **kwargs):
        super().__init__(*args, **kwargs)

        cfg = dict(imd_cfg or {})
        self.backward_iters = int(cfg.get("backward_iters", 10))
        self.damping_factor = float(cfg.get("damping_factor", 0.5))

        # Runtime parameters passed from forward (keeps signature backward compatible).
        self.current_forward_iters = int(cfg.get("current_forward_iters_default", 30))
        self.current_min_iters = int(cfg.get("current_min_iters_default", 5))
        self.current_tol_rel_flow = float(cfg.get("current_tol_rel_flow_default", 1e-4))
        self.current_use_early_stop = bool(cfg.get("preserve_early_stop", True))

        self._imd_last_info = {
            "iterations": 0.0,
            "converged_ratio": 0.0,
            "final_gap": 0.0,
            "implicit_grad": True,
        }

    def _link_flows_from_sigma(self, sigma_flat: torch.Tensor, q_flat: torch.Tensor) -> torch.Tensor:
        route_flows = sigma_flat * q_flat
        return torch.sparse.mm(self.delta, route_flows.t()).t()

    def _fixed_point_step(self, sigma_current: torch.Tensor, od_demand: torch.Tensor, q_flat: torch.Tensor) -> torch.Tensor:
        bsz = sigma_current.shape[0]
        route_flows = sigma_current * q_flat
        link_flows = torch.sparse.mm(self.delta, route_flows.t()).t()
        link_times = self.vdf(link_flows)
        perceived = self._compute_perceived_cost(od_demand, route_flows, link_flows, link_times)

        c3 = perceived.view(bsz, self.num_od, self.k_paths)
        valid = self.validity_mask.unsqueeze(0).expand(bsz, -1, -1)
        c3_masked = torch.where(valid, c3, torch.full_like(c3, float("inf")))

        c_min = c3_masked.min(dim=2, keepdim=True).values
        c_shift = torch.where(valid, c3 - c_min, torch.zeros_like(c3))
        exp_term = torch.exp(torch.clamp(-self.current_eta * c_shift, min=-30.0, max=30.0))

        sigma3 = sigma_current.view(bsz, self.num_od, self.k_paths)
        sigma_next3 = sigma3 * exp_term
        sigma_next3 = self._normalize_sigma(sigma_next3)
        return sigma_next3.reshape(bsz, self.num_routes)

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
        bsz = od_demand.shape[0]
        device = od_demand.device

        self.current_eta = float(eta)
        self.current_forward_iters = int(max_iters)
        self.current_min_iters = int(min_iters)
        self.current_tol_rel_flow = float(tol_rel_flow)
        self.current_use_early_stop = bool(use_early_stop)

        if warm_start and self.cached_sigma.shape[1] == self.num_routes and self.cached_sigma.shape[0] == 1:
            sigma_init = self.cached_sigma.expand(bsz, -1).to(device)
        else:
            sigma_init = self._uniform_sigma(bsz, device)

        sigma_init = self._normalize_sigma(sigma_init.view(bsz, self.num_od, self.k_paths)).reshape(bsz, self.num_routes)
        q_flat = od_demand.unsqueeze(-1).expand(-1, -1, self.k_paths).reshape(bsz, self.num_routes)

        module_params = tuple(self.parameters())
        sigma_star = IMDMirrorDescentFunction.apply(self, sigma_init, od_demand, q_flat, *module_params)

        route_flows = sigma_star * q_flat
        link_flows = torch.sparse.mm(self.delta, route_flows.t()).t()
        link_times = self.vdf(link_flows)
        route_costs = self._compute_perceived_cost(od_demand, route_flows, link_flows, link_times)

        self.cached_sigma = sigma_star.detach().mean(dim=0, keepdim=True)

        info = {
            "iterations": torch.tensor(self._imd_last_info["iterations"], device=device),
            "converged_ratio": torch.tensor(self._imd_last_info["converged_ratio"], device=device),
            "final_gap": torch.tensor(self._imd_last_info["final_gap"], device=device),
            "implicit_grad": torch.tensor(1.0, device=device),
        }
        return link_flows, route_flows, route_costs, info


class CyclicODModel(BaseCyclicODModel):
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
        super().__init__(
            num_links=num_links,
            num_od_pairs=num_od_pairs,
            architecture=architecture,
            delta_matrix=delta_matrix,
            route_validity_mask=route_validity_mask,
            od_pair_indices=od_pair_indices,
            t0=t0,
            capacity=capacity,
            lanes=lanes,
            num_link_groups=num_link_groups,
            link_group=link_group,
            vdf_config=vdf_config,
            **kwargs,
        )

        physics_cfg = kwargs.get("physics", {})
        if isinstance(physics_cfg, DictConfig):
            physics_cfg = dict(physics_cfg)

        imd_cfg = kwargs.get("imd", {})
        if isinstance(imd_cfg, DictConfig):
            imd_cfg = dict(imd_cfg)

        self.assignment = PhysicsAwareMirrorDescentAssignmentIMD(
            delta_matrix=delta_matrix,
            route_validity_mask=route_validity_mask,
            od_pair_indices=od_pair_indices,
            t0=t0,
            capacity=capacity,
            lanes=lanes,
            num_link_groups=num_link_groups,
            link_group=link_group,
            vdf_config=vdf_config,
            feature_dim=int(architecture.feature_dim),
            physics_cfg=physics_cfg,
            imd_cfg=imd_cfg,
        )

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        conv_info = output.get("convergence_info")
        if isinstance(conv_info, dict):
            mode = str(conv_info.get("mode", "data_driven"))
            conv_info["implicit_grad"] = mode == "physics"
            output["convergence_info"] = conv_info
        return output


LossIMD = Loss
TrainingDiagnosticianIMD = TrainingDiagnostician

# Explicit class alias used by Hydra target to make IMD variant intent clear.
CyclicODModelIMD = CyclicODModel
