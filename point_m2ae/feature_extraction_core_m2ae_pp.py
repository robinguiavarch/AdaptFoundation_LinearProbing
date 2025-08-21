"""
Point-M2AE feature extraction core module with corrected preprocessing.

Implements Point-M2AE encoder-only feature extraction with corrected v1 preprocessing
that preserves anatomical position while maintaining Point-M2AE distribution requirements.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

# Setup paths identical to test_model.py
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


class PointM2AEFeatureExtractor(torch.nn.Module):
    """
    Point-M2AE encoder-only feature extractor with corrected preprocessing.
    
    Implements hierarchical point cloud feature extraction with corrected v1 preprocessing
    that preserves anatomical position using grid center as origin and [-1, 1] distribution.
    
    Attributes:
        approach (str): Aggregation approach ('feat_mean')
        config (dict): Configuration dictionary from YAML
        device (torch.device): Computation device
        cfg: Model configuration object
        encoder: H_Encoder model
        groupers: Group modules for hierarchical processing
    """
    
    def __init__(self, approach: str, checkpoint_path: Path, config: dict, preprocessing_version: str = 'v1'):
        """
        Initialize Point-M2AE feature extractor with corrected preprocessing.
        
        Args:
            approach (str): Aggregation approach ('feat_mean')
            checkpoint_path (Path): Path to Point-M2AE checkpoint
            config (dict): Configuration dictionary from YAML
            preprocessing_version (str): Preprocessing version ('v1' or 'v2'). Defaults to 'v1'.
        """
        super().__init__()
        
        if approach not in ['feat_mean']:
            raise ValueError("approach must be 'feat_mean'")
        
        if preprocessing_version not in ['v1', 'v2']:
            raise ValueError("preprocessing_version must be 'v1' or 'v2'")
        
        self.approach = approach
        self.preprocessing_version = preprocessing_version
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create config object from YAML processing section
        processing_config = config['processing']
        self.cfg = self._create_config_object(processing_config)
        
        # Initialize encoder and groupers
        self.encoder = H_Encoder(self.cfg).to(self.device).eval()
        self.groupers = self._make_groupers(self.cfg).to(self.device).eval()
        
        # Load encoder weights
        self._load_encoder_weights(checkpoint_path)
    
    def _create_config_object(self, processing_config: dict):
        """
        Create config object from YAML processing section.
        
        Args:
            processing_config (dict): Processing configuration from YAML
            
        Returns:
            Config object with required attributes
        """
        class Config:
            pass
        
        cfg = Config()
        cfg.encoder_depths = processing_config['encoder_depths']
        cfg.encoder_dims = processing_config['encoder_dims']
        cfg.group_sizes = processing_config['group_sizes']
        cfg.num_groups = processing_config['num_groups']
        cfg.local_radius = processing_config['local_radius']
        cfg.drop_path_rate = processing_config['drop_path_rate']
        cfg.num_heads = processing_config['num_heads']
        
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
    
    def _preprocess_roi_to_points(self, roi_3d: np.ndarray) -> np.ndarray:
        """
        Convert binary ROI to normalized point cloud using specified preprocessing version.
        
        Args:
            roi_3d (np.ndarray): Binary ROI with shape (Z, Y, X)
            
        Returns:
            np.ndarray: Processed points
        """
        if self.preprocessing_version == 'v1':
            return self._preprocess_v1_fixed_normalization(roi_3d)
        elif self.preprocessing_version == 'v2':
            return self._preprocess_v2_topological_features(roi_3d)
    
    @torch.no_grad()
    def extract_features(self, roi_volume: np.ndarray) -> torch.Tensor:
        """
        Extract features from single ROI volume.
        
        Args:
            roi_volume (np.ndarray): Binary ROI volume with shape (Z, Y, X)
            
        Returns:
            torch.Tensor: Extracted features
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
        
        # Apply aggregation
        return x_vis.mean(1)
    
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
        Get output feature dimension.
        
        Returns:
            int: Feature dimension (384 for feat_mean)
        """
        return 384