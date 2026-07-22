import logging
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.components.models.CGAME_DataDriven.CGAME_DataDriven_Auditer import (
    CGAMEDataDrivenDiagnostician,
)
from src.components.models.CGAME_DataDriven.CGAME_DataDriven_Logistician import (
    CGAMEDataDrivenDelegator,
)
from src.contracts.runtime_contracts import ArtifactSchemaError, require_keys

logger = logging.getLogger(__name__)

# =============================================================================
# 1. NEURAL COMPONENTS (Encoder/Decoder/Matcher)
# =============================================================================


class ODEncoder(nn.Module):
    """
    Encode input vectors (flows or OD) into latent features.
    Deep architecture: [Input -> H -> H/2 -> Feature].
    """

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
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ODDecoder(nn.Module):
    """
    Decode latent vector into physical outputs (OD or flows).
    Deep architecture: [Feature -> H -> H/2 -> Output].
    """

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
            nn.Softplus(),
        )

    def forward(self, g_x: torch.Tensor) -> torch.Tensor:
        return self.network(g_x)


class GraphMatcher(nn.Module):
    """
    Graph matcher with learned attention and explicit update routine.

    - The update step is decoupled from forward.
    - The forward pass remains differentiable.
    """

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

        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, num_structures),
            nn.Softmax(dim=-1),
        )

    def _ensure_batch_dim(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            return h.unsqueeze(0)
        return h

    @torch.no_grad()
    def update(self, h_x: torch.Tensor, h_y: torch.Tensor):
        """Explicit update of M and V matrices."""
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
        """Apply the learned transformation."""
        h_x = self._ensure_batch_dim(h_x)

        attn_weights = self.attention_net(h_x)

        h_exp = h_x.unsqueeze(2)
        h_struct = h_exp * self.M.unsqueeze(0)
        h_weighted = h_struct * self.V.unsqueeze(0)

        g_x = (h_weighted * attn_weights.unsqueeze(1)).sum(dim=2)

        return g_x


# =============================================================================
# 2. MAIN MODEL (CyclicODModel - Data Driven)
# =============================================================================


class CyclicODModel(nn.Module):
    def __init__(
        self,
        num_links: int,
        num_od_pairs: int,
        architecture: DictConfig,
        diagnostics: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__()
        logger.info("INIT: CyclicODModel Data-Driven (Deep Encoders + Matcher).")

        arch = architecture
        feature_dim = arch.feature_dim
        num_structures = arch.num_structures

        h_enc = arch.hidden_dim_from_link
        h_dec = arch.hidden_dim_to_od
        h_bwd_enc = kwargs.get("hidden_dim_from_od", h_dec)
        h_bwd_dec = kwargs.get("hidden_dim_to_link", h_enc)

        dropout = arch.get("dropout", 0.1)

        self.register_buffer(
            "link_scale",
            torch.tensor(kwargs.get("link_scale", 1.0), dtype=torch.float32),
        )
        self.register_buffer(
            "od_scale",
            torch.tensor(kwargs.get("od_scale", 1.0), dtype=torch.float32),
        )

        self.f_encoder = ODEncoder(num_links, h_enc, feature_dim, dropout)
        self.f_decoder = ODDecoder(num_od_pairs, h_dec, feature_dim, dropout)

        self.b_encoder = ODEncoder(num_od_pairs, h_bwd_enc, feature_dim, dropout)
        self.b_decoder = ODDecoder(num_links, h_bwd_dec, feature_dim, dropout)

        self.matcher = GraphMatcher(feature_dim, num_structures)

        loss_cfg = kwargs.get("loss", {})
        if isinstance(loss_cfg, DictConfig):
            loss_cfg = dict(loss_cfg)
        else:
            loss_cfg = dict(loss_cfg) if isinstance(loss_cfg, dict) else {}
        loss_cfg.pop("_target_", None)

        link_scale_cfg = loss_cfg.pop("link_scale", None)
        od_scale_cfg = loss_cfg.pop("od_scale", None)

        if link_scale_cfg is None:
            link_scale_cfg = kwargs.get("link_scale", None)
            if link_scale_cfg is None:
                link_scale_cfg = 1.0
                logger.warning(
                    "link_scale was not provided in model loss config or kwargs. Falling back to 1.0."
                )

        if od_scale_cfg is None:
            od_scale_cfg = kwargs.get("od_scale", None)
            if od_scale_cfg is None:
                od_scale_cfg = 1.0
                logger.warning(
                    "od_scale was not provided in model loss config or kwargs. Falling back to 1.0."
                )

        self.link_scale.fill_(float(link_scale_cfg))
        self.od_scale.fill_(float(od_scale_cfg))

        self.loss_fn = Loss(
            link_scale=float(self.link_scale.detach().item()),
            od_scale=float(self.od_scale.detach().item()),
            **loss_cfg,
        )

        self.delegator = CGAMEDataDrivenDelegator(self)

        self.diagnostics_cfg = dict(diagnostics or {})
        self.diagnostician = None
        if self.diagnostics_cfg:
            diag_cfg = {
                key: value
                for key, value in self.diagnostics_cfg.items()
                if key not in {"_target_"}
            }
            self.diagnostician = CGAMEDataDrivenDiagnostician(**diag_cfg)

    def forward(
        self,
        observed_flows: torch.Tensor,
        flow_mask: Optional[torch.Tensor] = None,
        true_od_demand: Optional[torch.Tensor] = None,
        od_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

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
            "hx": hx,
            "gx": gx,
            "hy": hy,
            "gy": gy,
        }

        output["loss"] = self.loss_fn(
            predicted_flows=flow_hat,
            true_flows=observed_flows,
            flow_mask=flow_mask,
            predicted_od=od_hat,
            true_od=true_od_demand,
            od_mask=od_mask,
        )

        return output

    def get_evaluation_artifacts(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs_map = require_keys(
            outputs,
            ["reconstructed_flows", "estimated_demand"],
            context="CGAME_DataDriven.get_evaluation_artifacts outputs",
            exc_type=ArtifactSchemaError,
        )
        return {
            "pred_flows": outputs_map["reconstructed_flows"].detach().cpu(),
            "pred_od": outputs_map["estimated_demand"].detach().cpu(),
            "hx": outputs_map.get("hx"),
            "gx": outputs_map.get("gx"),
            "hy": outputs_map.get("hy"),
            "gy": outputs_map.get("gy"),
        }


# =============================================================================
# 3. LOSS FUNCTION (Universal Adapter)
# =============================================================================


class Loss(nn.Module):
    def __init__(self, link_scale=1.0, od_scale=1.0, w_flow=1.0, w_od=1.0, **kwargs):
        super().__init__()
        self.register_buffer("link_scale", torch.tensor(float(link_scale)))
        self.register_buffer("od_scale", torch.tensor(float(od_scale)))
        self.w_flow = w_flow
        self.w_od = w_od
        self.mse = nn.MSELoss()

    def forward(self, outputs=None, targets=None, **kwargs):
        """Universal adapter: accepts dictionaries or unpacked kwargs."""

        pred_flow = kwargs.get("predicted_flows")
        if pred_flow is None and outputs:
            pred_flow = outputs.get("reconstructed_flows")

        true_flow = kwargs.get("true_flows")
        if true_flow is None and targets:
            true_flow = targets.get("flows")

        flow_mask = kwargs.get("flow_mask")
        if flow_mask is None and targets:
            flow_mask = targets.get("flow_mask")

        if pred_flow is None or true_flow is None:
            return {
                "total_loss": torch.tensor(0.0, requires_grad=True, device=self.link_scale.device),
                "l_flow": torch.tensor(0.0, device=self.link_scale.device),
                "l_od": torch.tensor(0.0, device=self.link_scale.device),
            }

        if flow_mask is None:
            flow_mask = torch.ones_like(true_flow)

        scaled_pred_f = pred_flow / self.link_scale
        scaled_true_f = true_flow / self.link_scale
        loss_flow = (self.mse(scaled_pred_f, scaled_true_f) * flow_mask).sum() / (
            flow_mask.sum() + 1e-6
        )

        loss_od = torch.tensor(0.0, device=pred_flow.device)

        pred_od = kwargs.get("predicted_od")
        if pred_od is None:
            pred_od = kwargs.get("estimated_demand")
        if pred_od is None and outputs:
            pred_od = outputs.get("estimated_demand")

        true_od = kwargs.get("true_od")
        if true_od is None:
            true_od = kwargs.get("od")
        if true_od is None:
            true_od = kwargs.get("true_od_demand")
        if true_od is None and targets:
            true_od = targets.get("od")

        od_mask = kwargs.get("od_mask")
        if od_mask is None and targets:
            od_mask = targets.get("od_mask")

        if pred_od is not None and true_od is not None and od_mask is not None:
            if od_mask.sum() > 0:
                scaled_pred_od = pred_od / self.od_scale
                scaled_true_od = true_od / self.od_scale
                loss_od = (self.mse(scaled_pred_od, scaled_true_od) * od_mask).sum() / (
                    od_mask.sum() + 1e-6
                )

        total_loss = (self.w_flow * loss_flow) + (self.w_od * loss_od)

        return {"total_loss": total_loss, "l_flow": loss_flow, "l_od": loss_od}
