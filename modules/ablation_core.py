import os
import math
from typing import Optional, Dict
try:
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
except ImportError:
    plt = None
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .distinguishability_utils import ArnoldScrambler
from utils.bit_metrics import calculate_bcr

# Import evaluation metrics.
try:
    from utils.nc_metrics import calculate_collision_nc
except ImportError:
    calculate_collision_nc = None

# Public module interface.
__all__ = [
    'DAF', 'MS_CAM', 'AFF', 'iAFF',
    'FullResNet50Backbone',
    'WatermarkFullEncoder',
    'MoCoSinglePath',
    'ArnoldScrambler',
    'calculate_bcr',
    'plot_history',
    'evaluate_watermark_nc',
]

class FullResNet50Backbone(nn.Module):
    """Extract Stage 4 features with a ResNet-50 backbone."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        # Extract Stage 4 features.
        self.model = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),
        )
        self.focus_dim = 768
        # Project the 2,048 Stage 4 channels to the target dimension.
        self.align_focus = nn.Conv2d(2048, self.focus_dim, kernel_size=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        f_focus = self.align_focus(features[0])
        return f_focus

    
# iAFF components.
class DAF(nn.Module):
    """Fuse two feature maps by direct addition."""

    def __init__(self):
        super().__init__()

    def forward(self, x, residual):
        return x + residual


class MS_CAM(nn.Module):
    """Generate channel-attention weights with MS-CAM."""

    def __init__(self, channels: int = 64, r: int = 4):
        super().__init__()
        inter_channels = int(channels // r)

        # Preserve local spatial detail with pointwise convolutions.
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # Capture global context through global average pooling.
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        wei = self.sigmoid(xl + xg)
        return wei


class AFF(nn.Module):
    """Fuse two feature maps with one channel-attention stage."""

    def __init__(self, channels: int = 64, r: int = 4):
        super().__init__()
        self.ms_cam = MS_CAM(channels=channels, r=r)

    def forward(self, x, residual):
        xa = x + residual
        wei = self.ms_cam(xa)
        return 2 * x * wei + 2 * residual * (1 - wei)


class iAFF(nn.Module):
    """Fuse two feature maps with two iterative attention stages."""

    def __init__(self, channels: int = 64, r: int = 4):
        super().__init__()
        self.ms_cam1 = MS_CAM(channels=channels, r=r)
        self.ms_cam2 = MS_CAM(channels=channels, r=r)

    def forward(self, x, residual):
        xa = x + residual
        wei1 = self.ms_cam1(xa)
        xi = x * wei1 + residual * (1 - wei1)
        wei2 = self.ms_cam2(xi)
        return x * wei2 + residual * (1 - wei2)

class WatermarkFullEncoder(nn.Module):
    """Generate a soft-binarized watermark from Stage 4 features."""

    def __init__(self, dim_watermark: int = 1024, debug: bool = False, pretrained: bool = False):
        super().__init__()
        self.debug = debug

        self.backbone = FullResNet50Backbone(pretrained=pretrained)
        self.copyright_backbone = FullResNet50Backbone(pretrained=pretrained)

        self.focus_dim = self.backbone.focus_dim

        # Normalize both feature streams before attention-based fusion.
        self.bn_host_f = nn.BatchNorm2d(self.focus_dim)
        self.bn_copy_f = nn.BatchNorm2d(self.focus_dim)

        # 3. Fusion (Stage III)
        self.iAFF_fuser = iAFF(channels=self.focus_dim)

        # 4. Heads (Stage IV)
        self.head_main = nn.Sequential(
            nn.Linear(self.focus_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, dim_watermark),
        )
        self.sigmoid = nn.Sigmoid()

    def _print_debug_info(self, stage: int, **tensors):
        """Print tensor shapes and value ranges for debugging."""
        try:
            # Print only on device 0 to avoid duplicate multi-GPU logs.
            if torch.cuda.is_available() and torch.cuda.current_device() == 0:
                def get_stat(t):
                    if isinstance(t, torch.Tensor):
                        return f"{str(list(t.shape)):<20} Range: [{t.min().item():.3f}, {t.max().item():.3f}]"
                    return f"{str(t):<20} Range: N/A"

                if stage == 1:
                    print("\n--- [DEBUG] 1. Aligned Backbone Outputs ---")
                    print(f" > Host_Input     : {get_stat(tensors.get('Host_Input'))}")
                    print(f" > F_host         : {get_stat(tensors.get('F_host'))}")
                    print(f" > F_copy         : {get_stat(tensors.get('F_copy'))}")
                elif stage == 2:
                    print("\n--- [DEBUG] 2. Fusion & Pooling ---")
                    print(f" > F_fused        : {get_stat(tensors.get('F_fused'))}")
                    print(f" > V_Obj          : {get_stat(tensors.get('V_Obj'))}")
                elif stage == 3:
                    print("\n--- [DEBUG] 3. Final Outputs (Stage4-only) ---")
                    print(f" > Logits         : {get_stat(tensors.get('Logits'))}")
                    print(f" > V_Soft         : {get_stat(tensors.get('V_Soft'))}")
                    print(f" > V_Norm         : {get_stat(tensors.get('V_Norm'))}")
        except Exception:
            pass

    def forward(self, x, I_copy):
        # A. Extract Stage 4 features.
        f_h = self.backbone.forward_features(x)
        f_cp = self.copyright_backbone.forward_features(I_copy)

        # B. Normalize both feature streams.
        f_host = self.bn_host_f(f_h)
        f_copy = self.bn_copy_f(f_cp)

        if self.debug:
            self._print_debug_info(1, Host_Input=x, F_host=f_host, F_copy=f_copy)

        # C. Fuse features and apply global average pooling.
        f_fused = self.iAFF_fuser(f_host, f_copy)
        v_focus = F.adaptive_avg_pool2d(f_fused, (1, 1)).flatten(1)

        if self.debug:
            self._print_debug_info(2, F_fused=f_fused, V_Obj=v_focus)

        # D. Project to the watermark representation.
        logits_raw = self.head_main(v_focus)

        v_soft = self.sigmoid(logits_raw)
        v_norm = F.normalize(v_soft, dim=1)

        if self.debug:
            self._print_debug_info(3, Logits=logits_raw, V_Soft=v_soft, V_Norm=v_norm)

        return v_norm, v_soft

class MoCoSinglePath(nn.Module):
    """MoCo architecture with one queue and Stage 4 representations."""

    def __init__(
        self,
        base_encoder,
        dim_watermark: int = 1024,
        K: int = 4096,
        m: float = 0.999,
        T: float = 0.07,
        base_encoder_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.K, self.m, self.T = K, m, T
        base_kwargs = base_encoder_kwargs or {}
        # Initialize the query and key encoders.
        self.encoder_q = base_encoder(dim_watermark=dim_watermark, **base_kwargs)
        self.encoder_k = base_encoder(dim_watermark=dim_watermark, **base_kwargs)

        # Initialize query and key encoders with identical parameters.
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Store watermark representations in a single queue.
        self.register_buffer('queue', F.normalize(torch.randn(dim_watermark, K), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Update the key encoder using an exponential moving average."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        """Update the feature queue with the current key representations."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # Truncate a batch that would cross the queue boundary.
        if ptr + batch_size > self.K:
            batch_size = self.K - ptr
            keys = keys[:batch_size]
        
        self.queue[:, ptr:ptr + batch_size] = keys.T
        # Advance the queue pointer.
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    def forward(self, im_q, im_k, I_copy_q, I_copy_k):
        # Compute query features with gradients.
        q_norm, q_soft = self.encoder_q(im_q, I_copy_q)
        
        with torch.no_grad():
            # Update the key encoder.
            self._momentum_update_key_encoder()
            # Compute key features without gradients.
            k_norm, _ = self.encoder_k(im_k, I_copy_k)

        # Compute positive and negative contrastive logits.
        l_pos = torch.einsum('nc,nc->n', [q_norm, k_norm]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q_norm, self.queue.clone().detach()])
        logits_main = torch.cat([l_pos, l_neg], dim=1) / self.T

        return logits_main, q_soft, k_norm


def plot_history(history, save_dir: str, filename: str, title: str, ylabel: str):
    """Plot and save one training-history curve."""
    if not history:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history, marker='o', markersize=3, label=ylabel)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

@torch.no_grad()
def evaluate_watermark_nc(encoder_q, val_loader, cp_dataset, device, dim_watermark, arnold_iters):
    """Evaluate robustness BCR and batch-level discriminability NC."""
    if calculate_collision_nc is None:
        raise ImportError('Evaluation metrics are unavailable. Check the utils package.')

    encoder_q.eval()
    side_len = int(math.sqrt(dim_watermark))
    scrambler = ArnoldScrambler(size=side_len, iter_num=arnold_iters, device=device)
    
    # Metric accumulators.
    total_robust_bcr, total_coll_nc, n_batches = 0.0, 0.0, 0

    # Use the first copyright image as a fixed validation anchor.
    test_cp_loader = torch.utils.data.DataLoader(cp_dataset, batch_size=1, shuffle=False)
    I_copy_fixed = next(iter(test_cp_loader)).to(device)

    for batch in tqdm(val_loader, desc='Evaluation', leave=False):
        if len(batch) == 4:
            c_i, _, a_i, _ = [t.to(device) for t in batch]
        elif len(batch) == 2:
            c_i, a_i = [t.to(device) for t in batch]
        else:
            raise ValueError('Unexpected val_loader batch format.')

        B = c_i.size(0)
        I_cp = I_copy_fixed.expand(B, -1, -1, -1)
        
        # 1. Extract watermark representations.
        _, v_clean_soft = encoder_q(c_i, I_cp)
        _, v_atk_soft = encoder_q(a_i, I_cp)

        # 2. Binarize and scramble the watermarks.
        s_clean = scrambler.scramble((v_clean_soft > 0.5).float())
        s_atk = scrambler.scramble((v_atk_soft > 0.5).float())

        # 3. Compute the discriminability NC.
        current_coll_nc = calculate_collision_nc(s_clean).item()

        # 4. Accumulate metrics.
        total_robust_bcr += calculate_bcr(s_clean, s_atk)
        total_coll_nc += current_coll_nc
        n_batches += 1

    # Return robustness BCR and discriminability NC.
    return (
        total_robust_bcr / n_batches, 
        total_coll_nc / n_batches, 
    )
