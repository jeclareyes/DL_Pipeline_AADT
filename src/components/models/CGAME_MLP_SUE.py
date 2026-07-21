"""
Hybrid Deep Learning Model for Traffic Assignment with Partial Data.
Features: Deep Encoders, Attention-based Graph Matcher, and IFT-based SUE Validator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
from typing import Dict, Optional, Callable
import logging
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# 1. DATA-DRIVEN COMPONENTS (Encoder / Decoder / Matcher)
# =============================================================================

class ODEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, feature_dim),
            nn.LayerNorm(feature_dim) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ODDecoder(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus() 
        )

    def forward(self, g_x: torch.Tensor) -> torch.Tensor:
        return self.network(g_x)


class GraphMatcher(nn.Module):
    def __init__(self, feature_dim: int, num_structures: int,
                 lambda_m: float = 0.01, lambda_v: float = 0.01,
                 reg_strength: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_structures = num_structures
        self.lambda_m = lambda_m
        self.lambda_v = lambda_v
        self.reg_strength = reg_strength

        self.register_buffer('M', torch.randn(feature_dim, num_structures) * 0.1)
        self.register_buffer('V', torch.ones(1, num_structures))
        self.register_buffer('update_count', torch.tensor(0.0))

        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, num_structures),
            nn.Softmax(dim=-1)
        )

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

# =============================================================================
# 2. PHYSICS-DRIVEN COMPONENTS (Implicit SUE & Krylov Solver)
# =============================================================================

def pytorch_batched_gmres(linear_operator: Callable[[torch.Tensor], torch.Tensor],
                          b: torch.Tensor, max_iter: int = 15, tol: float = 1e-4, eps: float = 1e-8) -> torch.Tensor:
    batch_size, dim = b.shape
    device, dtype = b.device, b.dtype
    
    x = torch.zeros_like(b)
    r = b - linear_operator(x)
    r_norm = torch.norm(r, p=2, dim=1, keepdim=True)
    
    if torch.max(r_norm) < tol:
        return x

    V = torch.zeros(batch_size, dim, max_iter + 1, device=device, dtype=dtype)
    V[:, :, 0] = r / (r_norm + eps)
    H = torch.zeros(batch_size, max_iter + 1, max_iter, device=device, dtype=dtype)
    cs = torch.zeros(batch_size, max_iter, device=device, dtype=dtype)
    sn = torch.zeros(batch_size, max_iter, device=device, dtype=dtype)
    beta = torch.zeros(batch_size, max_iter + 1, 1, device=device, dtype=dtype)
    beta[:, 0, :] = r_norm

    for k in range(max_iter):
        v_k = V[:, :, k]
        w = linear_operator(v_k)
        
        for j in range(k + 1):
            v_j = V[:, :, j]
            h_jk = torch.sum(w * v_j, dim=1, keepdim=True)
            H[:, j, k] = h_jk.squeeze(1)
            w = w - h_jk * v_j
            
        h_next = torch.norm(w, p=2, dim=1)
        H[:, k + 1, k] = h_next
        V[:, :, k + 1] = w / (h_next.unsqueeze(1) + eps)
        
        for j in range(k):
            h_j_k = H[:, j, k].clone()
            h_jp1_k = H[:, j + 1, k].clone()
            H[:, j, k] = cs[:, j] * h_j_k + sn[:, j] * h_jp1_k
            H[:, j + 1, k] = -sn[:, j] * h_j_k + cs[:, j] * h_jp1_k
            
        h_k_k = H[:, k, k].clone()
        h_kp1_k = H[:, k + 1, k].clone()
        
        denom = torch.sqrt(h_k_k**2 + h_kp1_k**2 + eps)
        cs[:, k] = h_k_k / denom
        sn[:, k] = h_kp1_k / denom
        
        H[:, k, k] = cs[:, k] * h_k_k + sn[:, k] * h_kp1_k
        H[:, k + 1, k] = 0.0
        
        beta_k = beta[:, k, 0].clone()
        beta[:, k, 0] = cs[:, k] * beta_k
        beta[:, k + 1, 0] = -sn[:, k] * beta_k
        
        if torch.max(torch.abs(beta[:, k + 1, 0])) < tol:
            H = H[:, :k+1, :k+1]
            beta = beta[:, :k+1, :]
            V = V[:, :, :k+1]
            break
            
    y = torch.linalg.solve_triangular(H, beta, upper=True)
    x_k = torch.bmm(V, y).squeeze(2)
    x = x + x_k
    
    return x


class ImplicitSUEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, demands, cost_function, assignment_layer, max_iters, tol):
        with torch.no_grad():
            batch_size = demands.shape[0]
            zero_link_flows = torch.zeros_like(cost_function.t0).unsqueeze(0).expand(batch_size, -1)
            t0_costs = cost_function(zero_link_flows)
            if t0_costs.dim() == 1:
                t0_costs = t0_costs.unsqueeze(0).expand(batch_size, -1)
                
            flows, current_probs = assignment_layer(t0_costs, demands)
            
            converged = False
            for it in range(1, max_iters + 1):
                last_flows = flows.clone()
                costs = cost_function(flows)
                new_flows, current_probs = assignment_layer(costs, demands)
                
                alpha_msa = 1.0 / (it + 1)
                flows = flows + alpha_msa * (new_flows - flows)
                
                rel_error = torch.norm(flows - last_flows, p=2, dim=1) / torch.norm(last_flows, p=2, dim=1).clamp(min=1e-8)
                if torch.max(rel_error) <= tol:
                    converged = True
                    break

        ctx.save_for_backward(flows, demands)
        ctx.cost_function = cost_function
        ctx.assignment_layer = assignment_layer
        ctx.converged = converged 
        return flows

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.converged:
            return torch.zeros_like(ctx.saved_tensors[1]), None, None, None, None

        f_star, demands = ctx.saved_tensors
        cost_func = ctx.cost_function
        assign_layer = ctx.assignment_layer
        
        with torch.enable_grad():
            f_req_grad = f_star.detach().requires_grad_(True)
            d_req_grad = demands.detach().requires_grad_(True)
            costs = cost_func(f_req_grad)
            f_next, _ = assign_layer(costs, d_req_grad)
            
        def vjp_operator(v):
            jvp_f = torch.autograd.grad(outputs=f_next, inputs=f_req_grad, grad_outputs=v, retain_graph=True)[0]
            return v - jvp_f

        v_star = pytorch_batched_gmres(linear_operator=vjp_operator, b=grad_output, max_iter=15, tol=1e-3)
        grad_demands = torch.autograd.grad(outputs=f_next, inputs=d_req_grad, grad_outputs=v_star)[0]
        
        return grad_demands, None, None, None, None


class StochasticAssignmentLayer(nn.Module):
    def __init__(self, route_masks: torch.Tensor, mu: float = 1.0):
        super().__init__()
        self.num_od, self.k_paths, self.num_links = route_masks.shape

        if route_masks.is_sparse:
            route_masks = route_masks.coalesce()
            indices = route_masks.indices()
            values = route_masks.values()
            new_rows = indices[0] * self.k_paths + indices[1]
            new_cols = indices[2]  
            new_indices = torch.stack([new_rows, new_cols])

            self.register_buffer(
                'sparse_mask_2d',
                torch.sparse_coo_tensor(new_indices, values, size=(self.num_od * self.k_paths, self.num_links))
            )
        else:
            self.register_buffer('sparse_mask_2d', route_masks.reshape(-1, self.num_links).to_sparse())

        self.mu_raw = nn.Parameter(torch.tensor(float(mu)))

    @property
    def mu(self):
        return torch.clamp(F.softplus(self.mu_raw), min=0.1, max=10.0)

    def forward(self, link_costs: torch.Tensor, demands: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = link_costs.shape[0]
        costs_t = torch.transpose(link_costs, 0, 1)  
        route_costs_flat_t = torch.sparse.mm(self.sparse_mask_2d, costs_t)  
        route_costs = route_costs_flat_t.transpose(0, 1).view(batch_size, self.num_od, self.k_paths)

        min_costs, _ = torch.min(route_costs, dim=2, keepdim=True)
        stable_costs = route_costs - min_costs.detach()
        stable_costs = torch.clamp(stable_costs, max=50.0)

        exp_utility = torch.exp(-self.mu * stable_costs)
        sum_utility = torch.sum(exp_utility, dim=2, keepdim=True)
        route_probs = exp_utility / (sum_utility + 1e-9)

        route_flows = route_probs * demands.unsqueeze(2)
        route_flows_flat = route_flows.view(batch_size, -1)

        mask_t = self.sparse_mask_2d.t()  
        rf_t = torch.transpose(route_flows_flat, 0, 1)  
        link_flows_t = torch.sparse.mm(mask_t, rf_t)  
        link_flows = link_flows_t.transpose(0, 1)  

        return link_flows, route_probs


class AssignmentValidator(nn.Module):
    """ Strictly handles Physics (SUE). Neural routing is handled by CyclicODModel. """
    def __init__(self, num_links: int, t0: torch.Tensor, capacity: torch.Tensor,
                 lanes: torch.Tensor, route_masks: torch.Tensor, od_pair_indices: torch.Tensor,
                 num_od_pairs: int, num_link_groups: int, link_group: torch.Tensor,
                 vdf_config: dict, trips_scaler: float = 1.0,
                 max_iters: int = 20, convergence_threshold: float = 1e-4):
        super().__init__()
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.register_buffer('t0', t0)
        self.trips_scaler = trips_scaler

        self.cost_function = hydra.utils.instantiate(
            vdf_config, t0=t0, capacity=capacity, lanes=lanes,
            num_link_groups=num_link_groups, link_group=link_group, _recursive_=False
        )
        self.assignment_layer = StochasticAssignmentLayer(route_masks=route_masks, mu=1.0)

    def forward(self, estimated_demands: torch.Tensor, override_max_iters: int = None) -> tuple:
        current_max_iters = override_max_iters if override_max_iters is not None else self.max_iters
        real_estimated_demands = estimated_demands * self.trips_scaler

        with torch.no_grad():
            test_flows = ImplicitSUEFunction.apply(
                real_estimated_demands, self.cost_function, self.assignment_layer,
                current_max_iters, self.convergence_threshold
            )
            
        costs_check = self.cost_function(test_flows)
        next_flows_check, final_probs = self.assignment_layer(costs_check, real_estimated_demands)
        rel_gap = torch.norm(next_flows_check - test_flows, p=2, dim=1) / torch.norm(test_flows, p=2, dim=1).clamp(min=1e-8)
        
        is_converged = bool(torch.max(rel_gap) <= self.convergence_threshold * 10) 

        if is_converged:
            reconstructed_flows = ImplicitSUEFunction.apply(
                real_estimated_demands, self.cost_function, self.assignment_layer,
                current_max_iters, self.convergence_threshold
            )
            conv_info = {"converged": True, "iterations": current_max_iters, "implicit_grad": True, "mode": "physics"}
        else:
            fallback_steps = 5
            flows = test_flows.detach() 
            for it in range(fallback_steps):
                costs = self.cost_function(flows)
                new_flows, final_probs = self.assignment_layer(costs, real_estimated_demands)
                alpha_msa = 1.0 / (current_max_iters + it + 1)
                flows = flows + alpha_msa * (new_flows - flows)
                
            reconstructed_flows = flows
            conv_info = {"converged": False, "iterations": current_max_iters + fallback_steps, "implicit_grad": False, "mode": "physics_fallback"}

        learned_alpha = getattr(self.cost_function, 'get_alpha', lambda: None)()
        learned_beta = getattr(self.cost_function, 'get_beta', lambda: None)()

        return reconstructed_flows, learned_alpha, learned_beta, conv_info, final_probs


# =============================================================================
# 3. MASTER ORCHESTRATOR (CyclicODModel - True Hybridization)
# =============================================================================

class CyclicODModel(nn.Module):
    def __init__(self, num_links: int, num_od_pairs: int, architecture: DictConfig,
                 t0: torch.Tensor, capacity: torch.Tensor, lanes: torch.Tensor,
                 route_masks: torch.Tensor, od_pair_indices: torch.Tensor,
                 num_link_groups: int, link_group: torch.Tensor,
                 vdf_config: DictConfig, **kwargs):
        super().__init__()
        logger.info(f"INIT: Hybrid CyclicODModel (Deep Encoders + optional SUE Validator).")

        arch = architecture
        feature_dim = arch.feature_dim
        num_structures = arch.num_structures
        
        h_enc = arch.hidden_dim_from_link
        h_dec = arch.hidden_dim_to_od
        h_bwd_enc = kwargs.get('hidden_dim_from_od', h_dec)
        h_bwd_dec = kwargs.get('hidden_dim_to_link', h_enc)
        dropout = arch.get('dropout', 0.1)

        hybrid_cfg = kwargs.get('hybrid_assignment', {})
        self.hybrid_use_physics = bool(hybrid_cfg.get('use_physics', True))
        self.hybrid_warmup_epochs = int(hybrid_cfg.get('warmup_epochs', 50))

        self.register_buffer("link_scale", torch.tensor(kwargs.get('link_scale', 1.0), dtype=torch.float32))
        self.register_buffer("od_scale", torch.tensor(kwargs.get('od_scale', 1.0), dtype=torch.float32))

        # --- A. FORWARD NETWORK ---
        self.f_encoder = ODEncoder(num_links, h_enc, feature_dim, dropout)
        self.f_decoder = ODDecoder(num_od_pairs, h_dec, feature_dim, dropout)

        # --- B. BACKWARD NETWORK (Data-Driven Branch) ---
        self.b_encoder = ODEncoder(num_od_pairs, h_bwd_enc, feature_dim, dropout)
        self.b_decoder = ODDecoder(num_links, h_bwd_dec, feature_dim, dropout)

        # --- C. MATCHER ---
        self.matcher = GraphMatcher(feature_dim, num_structures)

        # --- D. SUE VALIDATOR (Physics-Driven Branch, optional) ---
        self.validator = None
        if self.hybrid_use_physics:
            max_iters = kwargs.get('msa_convergence', {}).get('max_iters', 20)
            flow_tol = kwargs.get('msa_convergence', {}).get('flow_tol', 1e-4)

            self.validator = AssignmentValidator(
                num_links=num_links, t0=t0, capacity=capacity, lanes=lanes,
                route_masks=route_masks, od_pair_indices=od_pair_indices, num_od_pairs=num_od_pairs,
                num_link_groups=num_link_groups, link_group=link_group, vdf_config=vdf_config,
                trips_scaler=kwargs.get('od_scale', 1.0), max_iters=max_iters, convergence_threshold=flow_tol
            )

        loss_cfg = kwargs.get("loss", {})
        if isinstance(loss_cfg, DictConfig):
            loss_cfg = dict(loss_cfg)
        else:
            loss_cfg = dict(loss_cfg) if isinstance(loss_cfg, dict) else {}
        loss_cfg.pop("_target_", None)

        # Priority: loss YAML -> model kwargs -> safe fallback (1.0)
        link_scale_cfg = loss_cfg.pop("link_scale", None)
        od_scale_cfg = loss_cfg.pop("od_scale", None)

        if link_scale_cfg is None:
            link_scale_cfg = kwargs.get("link_scale", None)
            if link_scale_cfg is None:
                link_scale_cfg = 1.0
                logger.warning("link_scale was not provided in model loss config or kwargs. Falling back to 1.0.")

        if od_scale_cfg is None:
            od_scale_cfg = kwargs.get("od_scale", None)
            if od_scale_cfg is None:
                od_scale_cfg = 1.0
                logger.warning("od_scale was not provided in model loss config or kwargs. Falling back to 1.0.")

        self.link_scale.fill_(float(link_scale_cfg))
        self.od_scale.fill_(float(od_scale_cfg))

        self.loss_fn = Loss(
            link_scale=float(self.link_scale.detach().item()),
            od_scale=float(self.od_scale.detach().item()),
            **loss_cfg,
        )

    def _forward_data_driven(self,
                             observed_flows: torch.Tensor,
                             flow_mask: Optional[torch.Tensor] = None,
                             true_od_demand: Optional[torch.Tensor] = None,
                             od_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Data-driven path intentionally mirrors CGAME_DataDriven behavior."""
        if flow_mask is None:
            flow_mask = torch.ones_like(observed_flows)

        x_in = (observed_flows / self.link_scale) * flow_mask

        hx = self.f_encoder(x_in)
        gx = self.matcher(hx)
        raw_od_hat = self.f_decoder(gx)
        od_hat = raw_od_hat * self.od_scale

        use_teacher_forcing = self.training and (true_od_demand is not None)
        if use_teacher_forcing:
            true_od_norm = true_od_demand / self.od_scale
            if od_mask is not None:
                od_input_mix = (true_od_norm * od_mask) + (raw_od_hat.detach() * (1.0 - od_mask))
            else:
                od_input_mix = true_od_norm
        else:
            od_input_mix = raw_od_hat

        hy = self.b_encoder(od_input_mix)
        gy = self.matcher(hy)
        raw_flow_hat = self.b_decoder(gy)
        flow_hat = raw_flow_hat * self.link_scale

        if self.training:
            self.matcher.update(hx.detach(), hy.detach())

        output = {
            "estimated_demand": od_hat,
            "reconstructed_flows": flow_hat,
            "convergence_info": {"converged": True, "iterations": 0, "implicit_grad": False, "mode": "data_driven"},
            "learned_alpha": None,
            "learned_beta": None,
            "route_probs": None,
            "hx": hx, "gx": gx,
            "hy": hy, "gy": gy
        }

        if true_od_demand is not None:
            output["loss"] = self.loss_fn(
                predicted_flows=flow_hat,
                true_flows=observed_flows,
                flow_mask=flow_mask,
                predicted_od=od_hat,
                true_od=true_od_demand,
                od_mask=od_mask,
                learned_alpha=output.get("learned_alpha"),
                learned_beta=output.get("learned_beta"),
            )

        return output

    def forward(self, observed_flows: torch.Tensor, flow_mask: Optional[torch.Tensor] = None,
                true_od_demand: Optional[torch.Tensor] = None, od_mask: Optional[torch.Tensor] = None,
                warmup: bool = False, current_epoch: Optional[int] = None,
                current_iter_count: Optional[int] = None, **kwargs) -> Dict[str, torch.Tensor]:

        # If physics is disabled, behave like CGAME_DataDriven.
        if not self.hybrid_use_physics:
            return self._forward_data_driven(
                observed_flows=observed_flows,
                flow_mask=flow_mask,
                true_od_demand=true_od_demand,
                od_mask=od_mask
            )

        is_batched = observed_flows.dim() == 2
        if not is_batched:
            observed_flows = observed_flows.unsqueeze(0)
            if flow_mask is not None: flow_mask = flow_mask.unsqueeze(0)
            if true_od_demand is not None: true_od_demand = true_od_demand.unsqueeze(0)
            if od_mask is not None: od_mask = od_mask.unsqueeze(0)

        if flow_mask is None: flow_mask = torch.ones_like(observed_flows)

        # 1. FORWARD PASS
        x_in = (observed_flows / self.link_scale) * flow_mask
        hx = self.f_encoder(x_in)
        gx = self.matcher(hx)      
        raw_od_hat = self.f_decoder(gx)
        od_hat = raw_od_hat * self.od_scale

        # 2. MATCHING & LATENT CALIBRATION (Always runs to keep spaces aligned)
        use_teacher_forcing = self.training and (true_od_demand is not None)
        if use_teacher_forcing:
            true_od_norm = true_od_demand / self.od_scale
            if od_mask is not None:
                od_input_mix = (true_od_norm * od_mask) + (raw_od_hat.detach() * (1.0 - od_mask))
            else:
                od_input_mix = true_od_norm
        else:
            od_input_mix = raw_od_hat

        hy = self.b_encoder(od_input_mix)
        gy = self.matcher(hy)      

        if self.training:
            self.matcher.update(hx.detach(), hy.detach())

        # 3. HYBRID DECISION: Neural vs Physics
        epoch_in_warmup = (current_epoch is not None) and (current_epoch < self.hybrid_warmup_epochs)
        use_physics = not (epoch_in_warmup or warmup)

        if not use_physics:
            # DATA-DRIVEN BRANCH
            raw_flow_hat = self.b_decoder(gy)
            flow_hat = raw_flow_hat * self.link_scale
            
            conv_info = {"converged": True, "iterations": 0, "implicit_grad": False, "mode": "neural_warmup"}
            learned_alpha, learned_beta, route_probs = None, None, None
        else:
            # PHYSICS-DRIVEN BRANCH (SUE)
            # SUE expects demands in REAL scale, not normalized.
            if self.validator is None:
                raise RuntimeError("SUE validator is not initialized while use_physics is enabled.")
            flow_hat, learned_alpha, learned_beta, conv_info, route_probs = self.validator(
                od_hat, override_max_iters=current_iter_count
            )

        if not is_batched:
            od_hat = od_hat.squeeze(0)
            flow_hat = flow_hat.squeeze(0)
            if route_probs is not None: route_probs = route_probs.squeeze(0)

        output = {
            "estimated_demand": od_hat,
            "reconstructed_flows": flow_hat,
            "convergence_info": conv_info,
            "learned_alpha": learned_alpha,
            "learned_beta": learned_beta,
            "route_probs": route_probs,
            "hx": hx, "gx": gx,
            "hy": hy, "gy": gy
        }

        if true_od_demand is not None:
            output["loss"] = self.loss_fn(
                predicted_flows=flow_hat,
                true_flows=observed_flows,
                flow_mask=flow_mask,
                predicted_od=od_hat,
                true_od=true_od_demand,
                od_mask=od_mask,
                learned_alpha=learned_alpha,
                learned_beta=learned_beta,
            )

        return output

    def get_evaluation_artifacts(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert isinstance(outputs, dict), "outputs must be a dict"
        assert "reconstructed_flows" in outputs, "Missing key 'reconstructed_flows'"
        assert "estimated_demand" in outputs, "Missing key 'estimated_demand'"
        return {
            "pred_flows": outputs["reconstructed_flows"].detach().cpu(),
            "pred_od": outputs["estimated_demand"].detach().cpu(),
            "convergence": outputs.get("convergence_info", {}),
            "learned_alpha": outputs.get("learned_alpha"),
            "learned_beta": outputs.get("learned_beta"),
            "route_probs": outputs.get("route_probs"),
        }

# =============================================================================
# 3. FUNCIÓN DE PÉRDIDA COMPATIBLE CON TRAINER.PY
# =============================================================================

class Loss(nn.Module):
    def __init__(self,
                 link_scale: float = 1.0,
                 od_scale: float = 1.0,
                 w_flow: float = 1.0,
                 w_od: float = 1.0,
                 w_demand_reg: float = 1e-6,
                 t0_costs: Optional[torch.Tensor] = None):
        super().__init__()
        # Registramos los escaladores para asegurar que los errores estén en el rango [0, 1]
        self.register_buffer("link_scale", torch.tensor(float(link_scale)))
        self.register_buffer("od_scale", torch.tensor(float(od_scale)))

        self.register_buffer("w_flow", torch.tensor(float(w_flow)))
        self.register_buffer("w_od", torch.tensor(float(w_od)))
        self.register_buffer("w_demand_reg", torch.tensor(float(w_demand_reg)))

        self.mse = nn.MSELoss(reduction="none")

        ### TODO
        # PRIOR GRAVITACIONAL (El Ancla)
        # Si no se provee, usamos un prior uniforme para evitar el cero absoluto
        if t0_costs is not None:
            # Opción más robusta numéricamente que 1/x^2
            # Normalizamos costos para evitar exponentes gigantes
            norm_costs = t0_costs / (t0_costs.mean() + 1e-6)
            gravity = torch.exp(-2.0 * norm_costs)  # Beta aprox 2.0
            self.register_buffer("prior_demand", gravity / gravity.mean())
        else:
            self.register_buffer("prior_demand", None)
        ### TODO

    def forward(
            self,
            predicted_flows: torch.Tensor,
            true_flows: torch.Tensor,
            flow_mask: torch.Tensor,
            predicted_od: torch.Tensor,
            true_od: torch.Tensor,
            od_mask: torch.Tensor,
            **kwargs
    ) -> Dict[str, torch.Tensor]:

        link_scale, od_scale = self.link_scale,  self.od_scale
        scale_loss = True
        if not scale_loss:
            link_scale = torch.tensor(1.0, device=predicted_flows.device)
            od_scale = torch.tensor(1.0, device=predicted_od.device)

        # --- 1. NORMALIZACIÓN DE ERROR DE FLUJO ---
        # Escalamos ambos tensores antes del MSE para que el error sea relativo a la capacidad.
        # Loss = MSE(f_pred / scale, f_true / scale)
        scaled_pred_flow = predicted_flows / link_scale
        scaled_true_flow = true_flows / link_scale

        l_flow_raw = self.mse(scaled_pred_flow, scaled_true_flow)
        l_flow = (l_flow_raw * flow_mask).sum() / flow_mask.sum().clamp(min=1.0)

        # --- 2. NORMALIZACIÓN DE ERROR DE OD ---
        # Hacemos lo mismo con la demanda para que el gradiente sea comparable al de flujo.
        if od_mask.any():
            scaled_pred_od = predicted_od / od_scale
            scaled_true_od = true_od / od_scale
            l_od_raw = self.mse(scaled_pred_od, scaled_true_od)
            l_od = (l_od_raw * od_mask).sum() / od_mask.sum().clamp(min=1.0)
        else:
            l_od = torch.tensor(0.0, device=predicted_od.device)

        # --- 3. REGULARIZACIÓN DE DEMANDA DESCONOCIDA ---
        # La regularización también debe ocurrir en el espacio escalado.
        unknown_mask = 1.0 - od_mask
        # l_demand_reg = ((predicted_od / self.od_scale) * unknown_mask).pow(2).mean() TODO

        # Normalizamos la predicción para compararla con el prior
        pred_norm = predicted_od / od_scale

        if self.prior_demand is not None:
            # Opción A: Gravity Regularization (Evita que mueran en 0 y evita explosiones locas)
            # Empujamos la demanda desconocida hacia el modelo de gravedad, no hacia cero.
            target = self.prior_demand

            # Ajustamos la magnitud del target para que coincida con la predicción actual
            # (queremos copiar la FORMA del gravity model, no necesariamente su magnitud exacta)
            with torch.no_grad():
                scale_factor = pred_norm.mean() / target.mean().clamp(min=1e-6)
                target_scaled = target * scale_factor

            l_demand_reg = (self.mse(pred_norm, target_scaled) * unknown_mask).sum() / unknown_mask.sum().clamp(min=1.0)
        else:
            # Opción B (Fallback): Regularización L2 Suave (hacia la media, no hacia cero)
            # Penalizamos desviarse de la media del batch actual
            batch_mean = pred_norm.detach().mean()
            l_demand_reg = ((pred_norm - batch_mean) * unknown_mask).pow(2).mean()

        # Pérdida Total Balanceada
        total = (self.w_flow * l_flow) + (self.w_od * l_od) + (self.w_demand_reg * l_demand_reg)

        return {
            "total_loss": total,
            "l_flow": l_flow,
            "l_od": l_od,
            "l_demand_reg": l_demand_reg,
            "flow_coverage": flow_mask.float().mean().item(),
            "od_coverage": od_mask.float().mean().item()
        }

##%

import torch
import matplotlib.pyplot as plt
from collections import deque, defaultdict


class CGAMEDiagnosticTool:
    def __init__(self, window_size=50):
        self.window_size = window_size

        # --- 1. Historial de Series Temporales (Moving Averages / Trends) ---
        self.history = {
            # Estructurales (Originales)
            'latent_sim': [],
            'v_entropy': [],
            'hx_norm_ma': deque(maxlen=window_size),
            's_iters': [],
            'route_entropy': [],

            # Diagnóstico 1: Loss Dynamics
            'loss_flow': [],
            'loss_od': [],
            'loss_reg': [],

            # Diagnóstico 4: Salud del Gradiente
            'grad_enc': [],
            'grad_match': [],
            'grad_dec': [],

            # Diagnóstico 5: GRADIENTE
            'grad_norm': defaultdict(list),  # Potencia bruta del gradiente
            'update_ratio': defaultdict(list),  # Relación (Gradiente / Peso)
            'sparsity': defaultdict(list),  # % de Gradientes Cero (Neuronas muertas)
            'flow_health': [],              # Ratio Decoder/Encoder (Global)

        }

        # --- 2. Datos Acumulados para Distribuciones (Originales) ---
        self.dist_data = {
            'gx_vals': [],
            'gy_vals': [],
            'v_weights': []
        }

        # --- 3. Datos Snapshot (Para gráficos pesados: Residuals y OD Check) ---
        # Solo guardamos el último estado para no explotar la memoria
        self.snap_data = {
            'true_flows': None,
            'pred_flows': None,
            'true_od_masked': None,
            'pred_od_all': None
        }

        # --- NUEVO: EVOLUTION TRACKING (Spectral + Sentinels) ---
        # Definimos bins fijos logarítmicos para asegurar consistencia temporal
        # Rango: 0.1 a 50,000 (Ajustable según tu escala)
        self.evolution_bins = np.geomspace(0.1, 10000, 50)

        # Estructura para guardar: 'counts' (Matriz de densidad) y 'traces' (Líneas individuales)
        self.evo_data = {
            'flow_known': {'counts': [], 'traces': [], 'indices': None, 'label': 'Flow (Known)'},
            'flow_unknown': {'counts': [], 'traces': [], 'indices': None, 'label': 'Flow (Unknown)'},
            'od_known': {'counts': [], 'traces': [], 'indices': None, 'label': 'OD (Known)'},
            'od_unknown': {'counts': [], 'traces': [], 'indices': None, 'label': 'OD (Unknown)'}
        }

        # =====================================================================
        # NUEVO: PHYSICS TRACKING (V/C & VDF)
        # =====================================================================

        # 1. Congestión (V/C)
        # Guardaremos percentiles [25, 50, 75, 90, 99] por época
        # Estructura: self.vc_stats['Motorway'] = [[p25, p50... epoch0], [p25... epoch1]]
        self.vc_stats = defaultdict(list)
        self.link_types_map = None  # Se llenará en la primera iteración

        # 2. VDF Parameters
        # Estructura: self.vdf_history[group_id] = {'alpha': [], 'beta': []}
        self.vdf_history = defaultdict(lambda: {'alpha': [], 'beta': []})

        # =====================
        # BEGIN: GRADIENT ANALYSIS SETUP
        # =====================
        self.grad_history = defaultdict(lambda: defaultdict(list))
        self.epoch_indices = []
        # END: GRADIENT ANALYSIS SETUP
        # =========================

    @torch.no_grad()
    def update(self, outputs, model, loss_dict=None, targets=None, static_info=None):
        """
        Captures metrics from a single forward/backward pass.

        Args:
            outputs: Diccionario retornado por model()
            model: La instancia del modelo (para pesos y gradientes)
            loss_dict: (NUEVO) Diccionario retornado por criterion()
            targets: (NUEVO) Diccionario con {flows, mask, od, od_mask}
        """
        # =====================================================================
        # A. MÉTRICAS ORIGINALES (Latent, Structure, SUE)
        # =====================================================================
        gx = outputs.get('gx')
        hx = outputs.get('hx')
        est_demand = outputs.get('estimated_demand')

        # Validación de rangos
        if gx is not None and gx.dim() == 1: gx = gx.unsqueeze(0)
        if hx is not None and hx.dim() == 1: hx = hx.unsqueeze(0)
        if est_demand is not None and est_demand.dim() == 1: est_demand = est_demand.unsqueeze(0)

        # 1. Latent Alignment
        if est_demand is not None:
            # Replicar lógica del matcher para obtener gy
            est_demand_norm = est_demand / model.od_scale
            hy_pred = model.b_encoder(est_demand_norm)
            if hy_pred.dim() == 1: hy_pred = hy_pred.unsqueeze(0)
            gy_pred = model.matcher.apply(hy_pred)

            if gx is not None:
                cos_sim = torch.nn.functional.cosine_similarity(gx, gy_pred, dim=-1).mean()
                self.history['latent_sim'].append(cos_sim.item())

                # Samplear para histogramas (1 de cada 10 valores para ahorrar memoria)
                self.dist_data['gx_vals'].extend(gx.flatten()[::10].cpu().numpy())
                self.dist_data['gy_vals'].extend(gy_pred.flatten()[::10].cpu().numpy())

        # 2. Estabilidad de V
        V = model.matcher.V
        v_prob = torch.softmax(V, dim=-1)
        v_ent = -(v_prob * torch.log(v_prob + 1e-9)).sum()
        self.history['v_entropy'].append(v_ent.item())
        self.dist_data['v_weights'].extend(V.flatten().cpu().numpy())

        # 3. Estabilidad hx
        if hx is not None:
            self.history['hx_norm_ma'].append(torch.norm(hx, p=2, dim=-1).mean().item())

        # 4. SUE Iters & Route Entropy
        conv_info = outputs.get('convergence_info', {})
        self.history['s_iters'].append(conv_info.get('iterations', 0))

        route_probs = outputs.get('route_probs')
        if route_probs is not None:
            if route_probs.dim() == 2: route_probs = route_probs.unsqueeze(0)
            r_ent = -(route_probs * torch.log(route_probs + 1e-9)).sum(dim=-1).mean()
            self.history['route_entropy'].append(r_ent.item())

        # =====================================================================
        # B. DIAGNÓSTICO 1: LOSS DYNAMICS
        # =====================================================================
        if loss_dict is not None:
            self.history['loss_flow'].append(
                loss_dict.get('l_flow', 0).item() if hasattr(loss_dict.get('l_flow'), 'item') else 0)
            self.history['loss_od'].append(
                loss_dict.get('l_od', 0).item() if hasattr(loss_dict.get('l_od'), 'item') else 0)
            self.history['loss_reg'].append(
                loss_dict.get('l_demand_reg', 0).item() if hasattr(loss_dict.get('l_demand_reg'), 'item') else 0)

        # =====================================================================
        # C. DIAGNÓSTICO 4: SALUD DEL GRADIENTE (Gradient Norms)
        # =====================================================================
        # Nota: Esto debe ejecutarse después de loss.backward()

        def compute_grad_norm(module):
            total_norm = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item()
            return total_norm

        self.history['grad_enc'].append(compute_grad_norm(model.f_encoder))
        self.history['grad_match'].append(compute_grad_norm(model.matcher))
        self.history['grad_dec'].append(compute_grad_norm(model.f_decoder))

        # =====================================================================
        # D. SNAPSHOT DATA (Para Diag 2 y 3)
        # =====================================================================
        if targets is not None:
            # Snapshot para Residuals (Diag 3)
            # Guardamos flows solo donde hay máscara
            mask = targets['mask'].bool()
            y_true = targets['flows'][mask].detach().cpu().numpy()
            y_pred = outputs['reconstructed_flows'][mask].detach().cpu().numpy()

            self.snap_data['true_flows'] = y_true
            self.snap_data['pred_flows'] = y_pred

            # Snapshot para OD Distribution (Diag 2)
            # Guardamos OD real (solo máscara) y Predicho (todo)
            od_mask = targets['od_mask'].bool()
            if targets.get('od') is not None:
                self.snap_data['true_od_masked'] = targets['od'][od_mask].detach().cpu().numpy()

            if est_demand is not None:
                self.snap_data['pred_od_all'] = est_demand.detach().cpu().numpy().flatten()

        # =====================================================================
        # E. EVOLUTION TRACKING (Spectral Density + Sentinels)
        # =====================================================================
        if targets is not None and outputs.get('reconstructed_flows') is not None:
            # 1. Preparar Datos Crudos
            # Flujos
            pred_flows = outputs['reconstructed_flows'].detach().flatten()
            mask_flow = targets['mask'].detach().flatten().bool()

            # OD (Si existe estimación)
            pred_od = outputs['estimated_demand'].detach().flatten() if outputs.get(
                'estimated_demand') is not None else None
            mask_od = targets['od_mask'].detach().flatten().bool() if targets.get(
                'od_mask') is not None else None

            # 2. Definir Grupos
            groups = {}

            # Flow Known vs Unknown
            if len(pred_flows) == len(mask_flow):
                groups['flow_known'] = pred_flows[mask_flow].cpu().numpy()
                groups['flow_unknown'] = pred_flows[~mask_flow].cpu().numpy()

            # OD Known vs Unknown
            if pred_od is not None and mask_od is not None:
                groups['od_known'] = pred_od[mask_od].cpu().numpy()
                groups['od_unknown'] = pred_od[~mask_od].cpu().numpy()

            # 3. Procesar cada grupo (Binning + Sentinel Selection)
            for key, data_array in groups.items():
                if len(data_array) == 0: continue

                # A. Spectral Density (Histograma del momento)
                # Usamos los bins fijos definidos en __init__
                counts, _ = np.histogram(data_array, bins=self.evolution_bins)
                self.evo_data[key]['counts'].append(counts)

                # B. Sentinel Traces (Seleccionar índices solo en la primera vez)
                if self.evo_data[key]['indices'] is None:
                    # Estrategia de Selección: 1 Max, 1 Median, 1 Random
                    n_samples = len(data_array)
                    if n_samples > 0:
                        idx_max = np.argmax(data_array)
                        idx_med = np.argsort(data_array)[n_samples // 2]
                        idx_rnd = np.random.randint(0, n_samples)
                        # Guardamos los índices relativos a este subgrupo
                        self.evo_data[key]['indices'] = [idx_max, idx_med, idx_rnd]
                    else:
                        self.evo_data[key]['indices'] = []

                # Guardar los valores de los centinelas en esta época
                current_indices = self.evo_data[key]['indices']
                if current_indices:
                    sentinel_values = data_array[current_indices]
                    self.evo_data[key]['traces'].append(sentinel_values)

        # =====================================================================
        # F. PHYSICS TRACKING (V/C & VDF)
        # =====================================================================

        # 1. Configuración Estática (Solo primera vez)
        if self.link_types_map is None and static_info is not None:
            self.link_types_map = static_info.get('link_types')  # Array [N_Links] con strings tipo "Motorway"
            self.capacity_ref = static_info.get('capacity')  # Tensor/Array [N_Links]
            if isinstance(self.capacity_ref, torch.Tensor):
                self.capacity_ref = self.capacity_ref.detach().cpu().numpy()

        # 2. Calcular V/C Ratios del Batch Actual
        if self.link_types_map is not None and outputs.get('reconstructed_flows') is not None:
            flows = outputs['reconstructed_flows'].detach().cpu().numpy().flatten()

            # Evitar división por cero
            caps = self.capacity_ref
            vc_ratios = np.divide(flows, caps, out=np.zeros_like(flows), where=caps > 0.1)

            # A. Global V/C Stats
            pcts = np.percentile(vc_ratios, [25, 50, 75, 90, 99])
            self.vc_stats['Global'].append(pcts)

            # B. Disaggregated V/C Stats (Por Link Type)
            unique_types = np.unique(self.link_types_map)
            for l_type in unique_types:
                mask = (self.link_types_map == l_type)
                if np.any(mask):
                    vals = vc_ratios[mask]
                    type_pcts = np.percentile(vals, [25, 50, 75, 90, 99])
                    self.vc_stats[str(l_type)].append(type_pcts)

        # 3. VDF Parameters Tracking
        # Intentamos recuperar alphas/betas del output (si el modelo los retorna)
        # o del modelo directamente si son accesibles
        alphas = outputs.get('learned_alpha')
        betas = outputs.get('learned_beta')

        if alphas is not None and betas is not None:
            # Asumimos que alphas es [Num_Link_Groups] o escalar
            if isinstance(alphas, torch.Tensor): alphas = alphas.detach().cpu().numpy().flatten()
            if isinstance(betas, torch.Tensor): betas = betas.detach().cpu().numpy().flatten()

            # Si es escalar, convertir a lista
            if np.ndim(alphas) == 0: alphas = [alphas]
            if np.ndim(betas) == 0: betas = [betas]

            for i, (a, b) in enumerate(zip(alphas, betas)):
                self.vdf_history[i]['alpha'].append(a)
                self.vdf_history[i]['beta'].append(b)

    def finalize_and_plot(self, save_path="cgame_diagnostics_advanced.png"):
        """Generates a comprehensive diagnostic dashboard (3x4 Grid)."""

        # Aumentamos tamaño para acomodar 12 subplots
        fig, axes = plt.subplots(3, 4, figsize=(24, 15))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        axes = axes.flatten()  # Facilita indexar 0-11

        # --- ROW 1: DINÁMICA DE APRENDIZAJE ---

        # 1. Loss Dynamics (Log Scale) [Diag 1]
        ax = axes[0]
        ax.plot(self.history['loss_flow'], label='L_Flow', alpha=0.7)
        ax.plot(self.history['loss_od'], label='L_OD', alpha=0.7)
        ax.plot(self.history['loss_reg'], label='L_Reg', alpha=0.7, linestyle='--')
        ax.set_yscale('log')
        ax.set_title("Loss Components (Log Scale)")
        ax.legend()

        # 2. Gradient Norms [Diag 4]
        ax = axes[1]
        ax.plot(self.history['grad_enc'], label='Enc', color='blue', linewidth=1)
        ax.plot(self.history['grad_match'], label='Match', color='purple', linewidth=1)
        ax.plot(self.history['grad_dec'], label='Dec', color='red', linewidth=1)
        ax.set_title("Gradient Norms (L2)")
        ax.set_yscale('log')
        ax.legend()

        # 3. SUE Iterations
        ax = axes[2]
        ax.plot(self.history['s_iters'], color='green', alpha=0.6)
        ax.set_title("SUE MSA Iterations")

        # 4. Latent Alignment Trend
        ax = axes[3]
        ax.plot(self.history['latent_sim'], color='blue')
        ax.set_title("Latent Cosine Sim ($g_x, g_y$)")
        ax.set_ylim(0, 1)

        # --- ROW 2: ESTRUCTURA Y ENTROPÍA ---

        # 5. V Entropy
        ax = axes[4]
        ax.plot(self.history['v_entropy'], color='purple')
        ax.set_title("Matcher V Entropy")

        # 6. hx Norm
        ax = axes[5]
        ax.plot(list(self.history['hx_norm_ma']), color='red')
        ax.set_title(f"Mean $h_x$ Norm (MA)")

        # 7. Route Entropy
        ax = axes[6]
        ax.plot(self.history['route_entropy'], color='orange')
        ax.set_title("Route Prob Entropy")

        # 8. V Weights Dist
        ax = axes[7]
        ax.hist(self.dist_data['v_weights'], bins=30, color='orange')
        ax.set_title("V Weights Dist")

        # --- ROW 3: FÍSICA Y REALISMO (NUEVOS) ---

        # 9. Latent Values Dist
        ax = axes[8]
        ax.hist(self.dist_data['gx_vals'], bins=50, alpha=0.5, label='$g_x$', density=True)
        ax.hist(self.dist_data['gy_vals'], bins=50, alpha=0.5, label='$g_y$', density=True)
        ax.set_title("Latent Space Dist ($g_x$ vs $g_y$)")
        ax.legend()

        # 10. OD Realism Check (Log Scale) [Diag 2]
        ax = axes[9]
        if self.snap_data['pred_od_all'] is not None:
            # Plot Predicted (All)
            ax.hist(self.snap_data['pred_od_all'], bins=50, alpha=0.5, label='Pred (All)', color='red', log=True)
            # Plot True (Known)
            if self.snap_data['true_od_masked'] is not None:
                ax.hist(self.snap_data['true_od_masked'], bins=50, alpha=0.5, label='True (Known)', color='blue',
                        log=True)
            ax.set_title("OD Demand Distribution (Log Y)")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No Data", ha='center')

        # 11. Residuals vs Magnitude [Diag 3]
        ax = axes[10]
        if self.snap_data['true_flows'] is not None:
            y_true = self.snap_data['true_flows']
            y_pred = self.snap_data['pred_flows']
            residuals = y_pred - y_true

            # Scatter Plot
            ax.scatter(y_true, residuals, alpha=0.3, s=5)
            ax.axhline(0, color='k', linestyle='--', linewidth=1)
            ax.set_xlabel("True Flow")
            ax.set_ylabel("Residual (Pred - True)")
            ax.set_title("Residuals vs Magnitude")
        else:
            ax.text(0.5, 0.5, "No Data", ha='center')

        # 12. Empty Slot (Future Use)
        axes[11].axis('off')
        axes[11].text(0.5, 0.5, "CGAME Diagnostic Tool\nv2.0", ha='center', fontsize=12)

        plt.savefig(save_path)
        print(f"Diagnostics saved to {save_path}")

    def plot_evolution(self, save_path="cgame_evolution.png"):
        """
        Generates the Spectral Density Heatmaps with Sentinel Traces.
        Focuses on the temporal evolution of distributions.
        """
        import matplotlib.colors as mcolors
        from matplotlib.ticker import ScalarFormatter

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        # Definir orden de ploteo
        plot_keys = ['flow_known', 'flow_unknown', 'od_known', 'od_unknown']

        for i, key in enumerate(plot_keys):
            ax = axes[i]
            data_struct = self.evo_data[key]

            # Verificar si hay datos
            if not data_struct['counts']:
                ax.text(0.5, 0.5, "No Data Available", ha='center')
                continue

            # --- 1. PLOT SPECTRAL DENSITY (HEATMAP) ---
            # Convertir lista de conteos a matriz (Epochs x Bins)
            density_matrix = np.array(data_struct['counts']).T  # Transponer para (Y=Bins, X=Epochs)

            # 2. DATA CHECK: Calculate max before converting to NaN
            v_max = np.max(density_matrix)

            # If the matrix is all zeros (common when od_mask is 100% known), 
            # v_max will be 0, which breaks LogNorm.
            if v_max <= 0:
                ax.text(0.5, 0.5, f"No active data for {key}\n(All values are 0 or mask is empty)", 
                        ha='center', fontsize=12, color='gray')
                ax.set_title(data_struct['label'])
                continue

            # Reemplazar ceros con NaN para que el fondo sea transparente/blanco
            density_matrix = density_matrix.astype(float)
            density_matrix[density_matrix == 0] = np.nan

            epochs = np.arange(len(data_struct['counts']))

            # Pcolormesh con escala Logarítmica para el COLOR (Density)
            # Usamos los bins definidos en init para el eje Y
            # 'cmap' sugerido: 'viridis' o 'plasma' (oscuro es poco, brillante es mucho)
            mesh = ax.pcolormesh(
                epochs,
                self.evolution_bins[:-1],  # Eje Y (Bin Edges inferiores)
                density_matrix,
                norm=mcolors.LogNorm(vmin=1, vmax=np.nanmax(density_matrix)),
                cmap='magma_r',  # Invertido: Claro=Fondo, Oscuro=Alta Densidad
                shading='auto',
                alpha=0.9
            )

            # --- 2. PLOT SENTINEL TRACES (LINES) ---
            if data_struct['traces']:
                traces_matrix = np.array(data_struct['traces'])  # (Epochs x 3)

                # Plot Max (Red), Median (Blue), Random (Green)
                labels = ['Max (Start)', 'Median (Start)', 'Random']
                colors = ['red', 'blue', 'green']

                for t_idx in range(traces_matrix.shape[1]):
                    ax.plot(epochs, traces_matrix[:, t_idx],
                            color=colors[t_idx % 3],
                            linewidth=1.5,
                            linestyle='--',
                            label=labels[t_idx] if i == 0 else "")  # Leyenda solo en el primero

            # --- FORMATTING ---
            # Aseguramos que los números se vean como enteros naturales
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            formatter.set_useOffset(False)  # Evita el "+1eX" en el eje
            ax.yaxis.set_major_formatter(formatter)

            ax.set_title(data_struct['label'])
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Magnitude (Linear Scale)")
            ax.grid(True, ls="-", alpha=0.2)

            # Añadir colorbar pequeño
            cbar = plt.colorbar(mesh, ax=ax)
            cbar.set_label('Density (Count)')

        # Leyenda global para trazas
        fig.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.95))

        plt.suptitle("Temporal Evolution: Spectral Density & Sentinel Traces", fontsize=16, y=0.98)
        plt.savefig(save_path)
        print(f"Evolution diagnostics saved to {save_path}")


    def plot_physics(self, save_path="cgame_physics.png"):
        """
        Genera el dashboard físico: V/C Analysis y VDF Evolution.
        """
        import matplotlib.pyplot as plt

        # Determinar layout dinámico
        # Fila 1: V/C General + VDF Params
        # Filas siguientes: Small Multiples para V/C Types

        link_types = [k for k in self.vc_stats.keys() if k != 'Global']
        num_types = len(link_types)

        # Configurar figura:
        # Arriba: 3 columnas (V/C Global, Alpha Evol, Beta Evol)
        # Abajo: Grid para Link Types (ej. 3 columnas x N filas)

        rows_needed_types = (num_types + 2) // 3
        total_rows = 2 + rows_needed_types  # 1 fila resumen, 1 fila pain curves, N filas tipos

        fig = plt.figure(figsize=(20, 5 * total_rows))

        # --- ROW 1: RESUMEN GENERAL ---
        # 1. V/C Global
        ax1 = plt.subplot2grid((total_rows, 3), (0, 0))
        self._plot_vc_bands(ax1, self.vc_stats['Global'], "Global Network Congestion")

        # 2. VDF Parameters Evolution
        ax2 = plt.subplot2grid((total_rows, 3), (0, 1))
        for grp_id, hist in self.vdf_history.items():
            ax2.plot(hist['alpha'], label=f'Grp {grp_id}')
        ax2.set_title("VDF Alpha Evolution")
        ax2.set_xlabel("Epochs")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = plt.subplot2grid((total_rows, 3), (0, 2))
        for grp_id, hist in self.vdf_history.items():
            ax3.plot(hist['beta'], label=f'Grp {grp_id}')
        ax3.set_title("VDF Beta Evolution")
        ax3.set_xlabel("Epochs")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # --- ROW 2: PAIN CURVES (SNAPSHOTS) ---
        # Graficamos la curva de costo en Epoch 0, 50% y 100%
        ax_pain = plt.subplot2grid((total_rows, 3), (1, 0), colspan=3)
        self._plot_pain_curves(ax_pain)

        # --- ROWS 3+: V/C SMALL MULTIPLES ---
        for i, l_type in enumerate(link_types):
            row_idx = 2 + (i // 3)
            col_idx = i % 3
            ax = plt.subplot2grid((total_rows, 3), (row_idx, col_idx))
            self._plot_vc_bands(ax, self.vc_stats[l_type], f"Congestion: {l_type}")

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Physics diagnostics saved to {save_path}")

    def _plot_vc_bands(self, ax, stats_list, title):
        """Helper para pintar bandas de percentiles."""
        data = np.array(stats_list)  # [Epochs, 5] -> [25, 50, 75, 90, 99]
        epochs = np.arange(len(data))

        # P50 (Mediana)
        ax.plot(epochs, data[:, 1], color='black', linewidth=1.5, label='Median')

        # Banda IQR (25-75) - Zona Típica
        ax.fill_between(epochs, data[:, 0], data[:, 2], color='blue', alpha=0.2, label='IQR (25-75%)')

        # Banda Extrema (90-99) - Cuellos de Botella
        ax.fill_between(epochs, data[:, 3], data[:, 4], color='red', alpha=0.15, label='Stress (90-99%)')

        # Referencia Capacidad
        ax.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_title(title)
        ax.set_ylim(0, max(2.0, np.max(data) * 1.1))  # Limitar Y para ver detalle, pero permitir picos
        ax.set_ylabel("V/C Ratio")
        ax.grid(True, alpha=0.3)
        if "Global" in title: ax.legend(loc='upper left')

    def _plot_pain_curves(self, ax):
        """Dibuja la curva BPR en diferentes momentos del entrenamiento."""
        x = np.linspace(0, 1.5, 100)  # V/C de 0 a 1.5

        # Seleccionar snapshots: Inicio, Medio, Final
        epochs_recorded = len(self.vdf_history[0]['alpha'])
        if epochs_recorded < 2: return

        snapshots = [0, epochs_recorded // 2, epochs_recorded - 1]
        styles = [':', '--', '-']
        labels = ['Start', 'Mid', 'End']

        for grp_id, hist in self.vdf_history.items():
            # Solo graficar el primer grupo para no saturar, o iterar colores
            color = f"C{grp_id}"

            for shot, style, lbl in zip(snapshots, styles, labels):
                if shot < len(hist['alpha']):
                    a = hist['alpha'][shot]
                    b = hist['beta'][shot]

                    # BPR Formula: 1 + alpha * (x)^beta
                    y = 1.0 + a * np.power(x, b)

                    ax.plot(x, y, linestyle=style, color=color, linewidth=2,
                            label=f"Grp{grp_id} ({lbl}): a={a:.2f}, b={b:.2f}")

        ax.set_title("Evolution of Cost Function (Pain Curve)")
        ax.set_xlabel("V/C Ratio")
        ax.set_ylabel("Travel Time Multiplier (T/t0)")
        ax.axvline(1.0, color='k', linestyle='-', alpha=0.1)
        ax.grid(True)
        ax.legend()


    ### -----------------------------------
    # BEGIN: GRADIENT ANALYSIS

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    def _identify_block(self, param_name):
        """Clasifica los parámetros en órganos vitales."""
        if "f_encoder" in param_name: return "Forward Encoder"
        if "b_encoder" in param_name: return "Backward Encoder"
        if "matcher" in param_name: return "Graph Matcher"
        if "decoder" in param_name: return "Decoder"
        if "validator" in param_name: return "SUE Validator"  # Si tiene params aprendibles
        return "Other"

    def capture_gradient_history(self, model, epoch):
        """
        Toma una biopsia completa de los gradientes.
        Llamar justo después de loss.backward() y antes de optimizer.step().
        """
        self.epoch_indices.append(epoch)

        # Contenedores temporales para esta epoch
        block_norms = defaultdict(list)
        block_ratios = defaultdict(list)
        block_sparsity = defaultdict(list)

        for name, param in model.named_parameters():
            if param.grad is not None:
                block = self._identify_block(name)

                # 1. Norma del Gradiente (Potencia)
                g_norm = param.grad.data.norm(2).item()
                block_norms[block].append(g_norm)

                # 2. Ratio Peso/Actualización (Estabilidad)
                # ¿Qué tan grande es el empuje comparado con el tamaño del peso?
                p_norm = param.data.norm(2).item()
                if p_norm > 1e-9:
                    ratio = g_norm / p_norm
                else:
                    ratio = 0.0
                block_ratios[block].append(ratio)

                # 3. Sparsity (Necrosis)
                # Porcentaje de elementos en el tensor de gradiente que son exactamente cero
                n_zeros = torch.sum(param.grad.data == 0).item()
                n_elements = param.grad.data.numel()
                sparsity = (n_zeros / n_elements) * 100.0
                block_sparsity[block].append(sparsity)

        # Agregación por bloque (Promedio)
        target_blocks = ["Forward Encoder", "Graph Matcher", "Decoder", "Backward Encoder"]

        # Variables para flujo global
        norm_encoder = 0.0
        norm_decoder = 0.0

        for block in target_blocks:
            if block in block_norms and block_norms[block]:
                # Promedios
                avg_norm = np.mean(block_norms[block])
                avg_ratio = np.mean(block_ratios[block])
                avg_sparsity = np.mean(block_sparsity[block])

                # Guardar para el bloque específico
                self.history['grad_norm'][block].append(avg_norm)
                self.history['update_ratio'][block].append(avg_ratio)
                self.history['sparsity'][block].append(avg_sparsity)

                # Capturar valores clave para el diagnóstico de flujo
                if block == "Forward Encoder": norm_encoder = avg_norm
                if block == "Decoder": norm_decoder = avg_norm

            else:
                # Si el bloque no tiene params con gradiente (ej. Matcher sin pesos)
                # No guardamos ceros para no ensuciar las escalas logarítmicas
                pass

        # Calcular Salud del Flujo (Decoder -> Encoder)
        # Si ratio ~ 1.0, el gradiente pasa perfecto. Si ratio ~ 0.0, muere en el camino.
        if norm_decoder > 1e-9:
            flow_health = norm_encoder / norm_decoder
        else:
            flow_health = 0.0
        self.history['flow_health'].append(flow_health)

    def plot_gradient_health(self, save_path="gradient_health_report.png"):
        """
        Genera un panel de control de 3 niveles con diagnóstico textual.
        """
        if not self.epoch_indices:
            print("No hay datos de gradiente para graficar.")
            return

        epochs = self.epoch_indices
        blocks = ["Forward Encoder", "Graph Matcher", "Decoder"]
        colors = {"Forward Encoder": "#1f77b4", "Graph Matcher": "#ff7f0e", "Decoder": "#2ca02c",
                  "Backward Encoder": "#9467bd"}

        # Configurar figura: 3 filas
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        plt.subplots_adjust(bottom=0.15, hspace=0.3)

        # --- 1. Potencia (Norma L2) ---
        for block in blocks:
            if block in self.history['grad_norm']:
                ax1.plot(epochs, self.history['grad_norm'][block], label=block, color=colors.get(block, 'k'))
        ax1.set_yscale('log')
        ax1.set_title('A. Potencia del Gradiente (Norma L2) [Log Scale]')
        ax1.set_ylabel('||Grad||')
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend(loc='upper right')

        # --- 2. Estabilidad (Update Ratio) ---
        for block in blocks:
            if block in self.history['update_ratio']:
                data = self.history['update_ratio'][block]
                ax2.plot(epochs, data, label=block, color=colors.get(block, 'k'))

        # Zonas de referencia
        ax2.axhline(y=1e-3, color='gray', linestyle='--', alpha=0.5, label='Healthy Limit')
        ax2.axhline(y=1e-5, color='red', linestyle=':', alpha=0.5, label='Frozen Limit')

        ax2.set_yscale('log')
        ax2.set_title('B. Relación Peso/Actualización (||Grad|| / ||Weight||)')
        ax2.set_ylabel('Ratio')
        ax2.grid(True, which="both", alpha=0.3)

        # --- 3. Necrosis (Sparsity) ---
        for block in blocks:
            if block in self.history['sparsity']:
                ax3.plot(epochs, self.history['sparsity'][block], label=block, color=colors.get(block, 'k'))

        ax3.set_title('C. Necrosis Neuronal (% Gradientes Cero)')
        ax3.set_ylabel('% Muerto')
        ax3.set_xlabel('Epochs')
        ax3.set_ylim(-5, 105)
        ax3.grid(True, alpha=0.3)

        # --- DIAGNÓSTICO TEXTUAL ---
        report = self._generate_detailed_diagnosis()
        fig.text(0.5, 0.02, report, ha='center', va='bottom', fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#333"),
                 family='monospace')

        # Guardar
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Reporte forense guardado en: {save_path}")

    def _generate_diagnosis_text(self):
        """Analiza los últimos valores y redacta un diagnóstico."""
        diagnosis = ["--- REPORTE FORENSE DE GRADIENTES (ÚLTIMA EPOCH) ---"]

        # Obtener últimos valores
        last_norms = {b: self.grad_history['norm'][b][-1] if self.grad_history['norm'][b] else 0
                      for b in ["Forward Encoder", "Graph Matcher", "Decoder"]}

        # 1. Chequeo de Vanishing Gradient Global
        decoder_grad = last_norms["Decoder"]
        encoder_grad = last_norms["Forward Encoder"]

        if decoder_grad < 1e-7:
            diagnosis.append("CRÍTICO: Muerte Cerebral. El gradiente no llega ni al Decoder.")
            diagnosis.append("   -> Posible Causa: Loss Function desconectada o parámetros congelados.")

        # 2. Chequeo de Bloqueo en el Matcher (Cuello de Botella)
        elif decoder_grad > 1e-4 and encoder_grad < 1e-7:
            diagnosis.append("BLOQUEO: El Matcher está actuando como un tapón.")
            diagnosis.append(
                f"   -> Decoder recibe señal ({decoder_grad:.2e}), pero Encoder no ({encoder_grad:.2e}).")
            diagnosis.append("   -> El gradiente muere al intentar cruzar la atención o el SUE.")

        # 3. Chequeo de Gradiente Sano
        elif encoder_grad > 1e-5:
            diagnosis.append("SALUDABLE: El flujo de gradiente recorre toda la red.")

        # 4. Chequeo de Explosión
        if any(v > 100 for v in last_norms.values()):
            diagnosis.append("PELIGRO: Explosión de Gradiente detectada (>100).")
            diagnosis.append("   -> Recomendación: Aumentar Clipping o reducir Learning Rate.")

        return "\n".join(diagnosis)

    def _generate_detailed_diagnosis(self):
        """Analiza los datos acumulados para dar un veredicto."""
        try:
            last_flow = self.history['flow_health'][-1]
            last_decoder = self.history['grad_norm']['Decoder'][-1] if self.history['grad_norm']['Decoder'] else 0
            last_encoder = self.history['grad_norm']['Forward Encoder'][-1] if self.history['grad_norm'][
                'Forward Encoder'] else 0

            lines = ["--- DIAGNÓSTICO AUTOMÁTICO ---"]

            # 1. Chequeo de Salud Global (Decoder)
            if last_decoder < 1e-6:
                lines.append("PARO CARDÍACO: Gradiente en Decoder casi nulo.")
                lines.append("   -> Causa: Loss Function mal escalada o meseta plana.")
            elif last_decoder > 1000:
                lines.append("EXPLOSIÓN: Gradientes gigantescos en Decoder.")
                lines.append("   -> Acción: Activar 'scale_loss=True' y revisar 'link_scale'.")
            else:
                lines.append(f"Entrada Gradiente: Saludable ({last_decoder:.2e})")

            # 2. Chequeo de Flujo (Cuello de Botella)
            if last_decoder > 1e-5:
                if last_flow < 0.01:
                    lines.append("OBSTRUCCIÓN SEVERA: El gradiente no cruza al Encoder.")
                    lines.append(f"   -> Solo llega el {last_flow * 100:.4f}% de la señal.")
                    lines.append("   -> Culpable probable: Graph Matcher o SUE (Jacobiana cero).")
                elif last_flow < 0.5:
                    lines.append("PÉRDIDA PARCIAL: El gradiente se debilita en el camino.")
                else:
                    lines.append("FLUJO LIBRE: El gradiente viaja correctamente.")

            # 3. Chequeo de Estancamiento (Ratio)
            enc_ratio = self.history['update_ratio']['Forward Encoder'][-1] if self.history['update_ratio'][
                'Forward Encoder'] else 0
            if enc_ratio < 1e-6 and last_encoder > 0:
                lines.append("CONGELAMIENTO: Gradiente existe pero es muy débil para mover pesos.")
                lines.append("   -> Acción: Incrementar Learning Rate.")

            return "\n".join(lines)
        except Exception:
            return "Datos insuficientes para diagnóstico."

    # END: GRADIENTE ANALYSIS
    #----------------------------------------------
