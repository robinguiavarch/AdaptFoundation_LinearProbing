"""
Point-M2AE feature extraction core module with 50 configurations.

Implements Point-M2AE encoder-only feature extraction with 10 configurations
and 5 aggregation methods for ultra-fine topological detection optimization.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

# Setup paths identical to existing scripts
HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
PM2AE_DIR = HERE.parent / "Point-M2AE"

for p in (ROOT, PM2AE_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# KNN compatibility shim - CRITICAL: apply BEFORE importing Group
import importlib, inspect
mods = importlib.import_module('models.modules')

try:
    from knn_cuda import knn as _knn_native
    sig = inspect.signature(_knn_native)
    if len(sig.parameters) == 3:
        def _knn_compat(x, y, k, transpose_mode=True):
            return _knn_native(x, y, k)
        mods.knn = _knn_compat
except Exception:
    pass

# Point-M2AE modules - AFTER shim
from models.Point_M2AE_Finetune import H_Encoder
from models.modules import Group


class PointM2AEFeatureExtractorConfigs(torch.nn.Module):
    """
    Point-M2AE encoder-only feature extractor with 50 configurations.
    
    Implements hierarchical point cloud feature extraction with configurable
    parameters (num_groups, group_sizes, local_radius) and 5 aggregation methods
    for optimization on ultra-sparse anatomical data.
    
    Attributes:
        config_name (str): Configuration identifier (C1-C10)
        aggregation_name (str): Aggregation method identifier (A1-A5)
        config (dict): Full configuration dictionary from YAML
        device (torch.device): Computation device
        cfg: Model configuration object
        encoder: H_Encoder model
        groupers: Group modules for hierarchical processing
        aggregation_method: Selected aggregation function
        output_dim (int): Output feature dimension
    """
    
    def __init__(self, config_name: str, aggregation_name: str, checkpoint_path: Path, config: dict):
        """
        Initialize Point-M2AE feature extractor with specific configuration and aggregation.
        
        Args:
            config_name (str): Configuration name (C1-C10)
            aggregation_name (str): Aggregation method name (A1-A5)
            checkpoint_path (Path): Path to Point-M2AE checkpoint
            config (dict): Configuration dictionary from YAML
        """
        super().__init__()
        
        if config_name not in [f"C{i}" for i in range(1, 11)]:
            raise ValueError(f"config_name must be C1-C10, got {config_name}")
        
        if aggregation_name not in [f"A{i}" for i in range(1, 6)]:
            raise ValueError(f"aggregation_name must be A1-A5, got {aggregation_name}")
        
        self.config_name = config_name
        self.aggregation_name = aggregation_name
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get configuration parameters
        config_params = config['processing']['configs'][config_name]
        aggregation_params = config['aggregation_methods'][aggregation_name]
        
        # Create model configuration object
        self.cfg = self._create_config_object(config['processing'], config_params)
        
        # Initialize encoder and groupers
        self.encoder = H_Encoder(self.cfg).to(self.device).eval()
        self.groupers = self._make_groupers(self.cfg).to(self.device).eval()
        
        # Set aggregation method and output dimension
        self.aggregation_method = self._get_aggregation_method(aggregation_name)
        self.output_dim = aggregation_params['output_dim']
        
        # Load encoder weights
        self._load_encoder_weights(checkpoint_path)
    
    def _create_config_object(self, processing_config: dict, config_params: dict):
        """
        Create config object combining fixed and configurable parameters.
        
        Args:
            processing_config (dict): Base processing configuration
            config_params (dict): Specific configuration parameters
            
        Returns:
            Config object with required attributes
        """
        class Config:
            pass
        
        cfg = Config()
        # Fixed parameters (tied to pre-trained weights)
        cfg.encoder_depths = processing_config['encoder_depths']
        cfg.encoder_dims = processing_config['encoder_dims']
        cfg.drop_path_rate = processing_config['drop_path_rate']
        cfg.num_heads = processing_config['num_heads']
        
        # Configurable parameters
        cfg.num_groups = config_params['num_groups']
        cfg.group_sizes = config_params['group_sizes']
        cfg.local_radius = config_params['local_radius']
        
        return cfg
    
    def _make_groupers(self, cfg) -> torch.nn.ModuleList:
        """
        Build Group modules for KNN/FPS grouping.
        
        Args:
            cfg: Model configuration object
            
        Returns:
            torch.nn.ModuleList: List of Group modules
        """
        groupers = torch.nn.ModuleList()
        for ng, gs in zip(cfg.num_groups, cfg.group_sizes):
            groupers.append(Group(num_group=ng, group_size=gs))
        return groupers
    
    def _load_encoder_weights(self, checkpoint_path: Path):
        """
        Load encoder weights from checkpoint.
        
        Args:
            checkpoint_path (Path): Path to checkpoint file
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        obj = torch.load(str(checkpoint_path), map_location="cpu")
        
        # Extract state dict
        sd = None
        for key in ("state_dict", "base_model", "model", "module"):
            if isinstance(obj, dict) and isinstance(obj.get(key, None), dict):
                sd = obj[key]
                break
        if sd is None and isinstance(obj, dict):
            sd = obj
        
        # Extract encoder keys
        encoder_sd = {}
        for k, v in sd.items():
            if k.startswith("h_encoder."):
                encoder_sd[k.replace("h_encoder.", "")] = v
        
        self.encoder.load_state_dict(encoder_sd, strict=False)
    
    def _get_aggregation_method(self, aggregation_name: str):
        """
        Get aggregation method function based on name.
        
        Args:
            aggregation_name (str): Aggregation method name (A1-A5)
            
        Returns:
            Aggregation method function
        """
        method_map = {
            'A1': self._aggregate_features_mean,
            'A2': self._aggregate_features_mean_std_min_max,
            'A3': self._aggregate_features_multi_level,
            'A4': self._aggregate_features_adaptive,
            'A5': self._aggregate_features_attention
        }
        return method_map[aggregation_name]
    
    def _preprocess_roi_to_points(self, roi_3d: np.ndarray) -> np.ndarray:
        """
        Convert binary ROI to normalized point cloud with corrected preprocessing.
        
        Uses fixed normalization: center (15,19,11) + [-1,1] distribution
        preserving anatomical position with Point-M2AE compatibility.
        
        Args:
            roi_3d (np.ndarray): Binary ROI with shape (Z, Y, X)
            
        Returns:
            np.ndarray: Normalized points with shape (N, 3)
        """
        if roi_3d.ndim != 3:
            raise ValueError(f"ROI must be 3D, got shape {roi_3d.shape}")
        
        idx = np.argwhere(roi_3d == 1)
        if idx.size == 0:
            raise ValueError("ROI has 0 active voxels")
        
        # Fixed normalization: center (15,19,11) + [-1,1] distribution
        center = np.array([15.0, 19.0, 11.0])
        pts = (idx.astype(np.float32) - center) / center
        
        return pts
    
    def _aggregate_features_mean(self, x_vis: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Baseline mean pooling aggregation.
        
        Args:
            x_vis (torch.Tensor): Final level tokens [batch, 64, 384]
            
        Returns:
            torch.Tensor: Aggregated features [batch, 384]
        """
        return x_vis.mean(dim=1)
    
    def _aggregate_features_mean_std_min_max(self, x_vis: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Enhanced statistics aggregation (mean + std + min + max).
        
        Args:
            x_vis (torch.Tensor): Final level tokens [batch, 64, 384]
            
        Returns:
            torch.Tensor: Aggregated features [batch, 1536]
        """
        mean_feat = x_vis.mean(dim=1)  # [batch, 384]
        std_feat = x_vis.std(dim=1)    # [batch, 384]
        min_feat = x_vis.min(dim=1)[0] # [batch, 384]
        max_feat = x_vis.max(dim=1)[0] # [batch, 384]
        
        return torch.cat([mean_feat, std_feat, min_feat, max_feat], dim=1)  # [batch, 1536]
    
    def _aggregate_features_multi_level(self, x_vis: torch.Tensor, neighborhoods, centers, idxs, **kwargs) -> torch.Tensor:
        """
        Multi-level aggregation using levels 2 and 3 with PyTorch hooks.
        
        Args:
            x_vis (torch.Tensor): Final level tokens [batch, 64, 384]
            neighborhoods: Neighborhood data for re-processing
            centers: Center data for re-processing
            idxs: Index data for re-processing
            
        Returns:
            torch.Tensor: Aggregated features [batch, 576]
        """
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # Hook on level 2 (encoder_blocks[1])
        handle_lvl2 = self.encoder.encoder_blocks[1].register_forward_hook(get_activation('level2'))
        
        try:
            # Re-run forward pass to capture level 2
            x_vis_new = self.encoder(neighborhoods, centers, idxs, eval=True)
            
            # Get level 2 activations
            x_level2 = activations['level2']  # [batch, 256, 192]
            
            # Aggregate both levels
            feat_lvl2 = x_level2.mean(dim=1)  # [batch, 192]
            feat_lvl3 = x_vis_new.mean(dim=1) # [batch, 384]
            
            # Concatenate multi-level features
            return torch.cat([feat_lvl2, feat_lvl3], dim=1)  # [batch, 576]
            
        finally:
            # Clean up hook
            handle_lvl2.remove()
    
    def _aggregate_features_adaptive(self, x_vis: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Variance-based adaptive pooling aggregation.
        
        Args:
            x_vis (torch.Tensor): Final level tokens [batch, 64, 384]
            
        Returns:
            torch.Tensor: Aggregated features [batch, 384]
        """
        epsilon = 1e-6
        token_std = x_vis.std(dim=2) + epsilon  # [batch, 64]
        weights = token_std / token_std.sum(dim=1, keepdim=True)  # Normalize per batch
        
        # Weighted aggregation
        return (x_vis * weights.unsqueeze(2)).sum(dim=1)  # [batch, 384]
    
    def _aggregate_features_attention(self, x_vis: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Attention-weighted pooling aggregation using variance as importance.
        
        Args:
            x_vis (torch.Tensor): Final level tokens [batch, 64, 384]
            
        Returns:
            torch.Tensor: Aggregated features [batch, 384]
        """
        alpha = 4.0
        token_std = x_vis.std(dim=2)  # [batch, 64]
        attn_scores = torch.softmax(alpha * token_std, dim=1)  # [batch, 64]
        
        # Attention-weighted aggregation
        return (x_vis * attn_scores.unsqueeze(2)).sum(dim=1)  # [batch, 384]
    
    @torch.no_grad()
    def extract_features(self, roi_volume: np.ndarray) -> torch.Tensor:
        """
        Extract features from single ROI volume using configured method.
        
        Args:
            roi_volume (np.ndarray): Binary ROI volume with shape (Z, Y, X)
            
        Returns:
            torch.Tensor: Extracted features with dimension based on aggregation method
        """
        # Preprocess ROI to points
        pts_np = self._preprocess_roi_to_points(roi_volume)
        pts = torch.from_numpy(pts_np.copy()).to(torch.float32)
        pts = pts.unsqueeze(0).to(self.device).contiguous()
        
        # Hierarchical grouping
        neighborhoods, centers, idxs = [], [], []
        cur = pts
        
        for grouper in self.groupers:
            cur = cur.contiguous()
            nei, ctr, idx = grouper(cur)
            neighborhoods.append(nei)
            centers.append(ctr)
            idxs.append(idx)
            cur = ctr
        
        # Encoder forward
        x_vis = self.encoder(neighborhoods, centers, idxs, eval=True)
        
        # Apply configured aggregation method
        if self.aggregation_name == 'A3':
            # Multi-level needs additional parameters
            return self.aggregation_method(x_vis, neighborhoods, centers, idxs)
        else:
            return self.aggregation_method(x_vis)
    
    def extract_features_batch(self, volumes: torch.Tensor) -> torch.Tensor:
        """
        Extract features from multiple volumes.
        
        Args:
            volumes (torch.Tensor): Multiple volumes with shape (N, Z, Y, X)
            
        Returns:
            torch.Tensor: Features for all volumes
        """
        all_features = []
        
        for i in range(volumes.shape[0]):
            roi_volume = volumes[i].numpy()
            features = self.extract_features(roi_volume)
            all_features.append(features.cpu())
        
        return torch.cat(all_features, dim=0)
    
    def get_output_dim(self) -> int:
        """
        Get output feature dimension based on aggregation method.
        
        Returns:
            int: Feature dimension
        """
        return self.output_dim
    
    def get_config_info(self) -> dict:
        """
        Get configuration information for current instance.
        
        Returns:
            dict: Configuration details
        """
        config_params = self.config['processing']['configs'][self.config_name]
        aggregation_params = self.config['aggregation_methods'][self.aggregation_name]
        
        return {
            'config_name': self.config_name,
            'aggregation_name': self.aggregation_name,
            'num_groups': config_params['num_groups'],
            'group_sizes': config_params['group_sizes'],
            'local_radius': config_params['local_radius'],
            'strategy': config_params['strategy'],
            'aggregation_description': aggregation_params['description'],
            'output_dim': self.output_dim
        }