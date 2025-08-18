"""
Point-M2AE feature extraction core module with optimized preprocessing.

Implements Point-M2AE encoder-only feature extraction with two preprocessing approaches:
v1 (fixed normalization) and v2 (topological features) for AdaptFoundation project.
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
    Point-M2AE encoder-only feature extractor with optimized preprocessing.
    
    Implements hierarchical point cloud feature extraction with two preprocessing approaches:
    v1 (fixed normalization preserving anatomical position) and v2 (topological features).
    
    Attributes:
        approach (str): Aggregation approach ('feat_mean_max')
        preprocessing_version (str): Preprocessing version ('v1' or 'v2')
        config (dict): Configuration dictionary from YAML
        device (torch.device): Computation device
        cfg: Model configuration object
        encoder: H_Encoder model
        groupers: Group modules for hierarchical processing
    """
    
    def __init__(self, approach: str, checkpoint_path: Path, config: dict, preprocessing_version: str = 'v1'):
        """
        Initialize Point-M2AE feature extractor with preprocessing version.
        
        Args:
            approach (str): Aggregation approach ('feat_mean_max')
            checkpoint_path (Path): Path to Point-M2AE checkpoint
            config (dict): Configuration dictionary from YAML
            preprocessing_version (str): Preprocessing version ('v1' or 'v2'). Defaults to 'v1'.
        """
        super().__init__()
        
        if approach not in ['feat_mean_max']:
            raise ValueError("approach must be 'feat_mean_max'")
        
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
    
    def _preprocess_v1_fixed_normalization(self, roi_3d: np.ndarray) -> np.ndarray:
        """
        Preprocessing v1: Fixed normalization preserving anatomical position.
        
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
        
        # Fixed normalization by grid dimensions
        pts = idx.astype(np.float32) / np.array([29.0, 37.0, 21.0])
        
        return pts
    
    def _preprocess_v2_topological_features(self, roi_3d: np.ndarray) -> np.ndarray:
        """
        Preprocessing v2: Fixed normalization with topological features.
        
        Args:
            roi_3d (np.ndarray): Binary ROI with shape (Z, Y, X)
            
        Returns:
            np.ndarray: Enriched features with shape (N, 7)
        """
        if roi_3d.ndim != 3:
            raise ValueError(f"ROI must be 3D, got shape {roi_3d.shape}")
        
        idx = np.argwhere(roi_3d == 1)
        if idx.size == 0:
            raise ValueError("ROI has 0 active voxels")
        
        enriched_features = []
        for pt in idx:
            x, y, z = int(pt[0]), int(pt[1]), int(pt[2])
            
            # Fixed normalization
            pos_norm = pt.astype(np.float32) / np.array([29.0, 37.0, 21.0])
            
            # Topological features
            nb_neighbors = self._count_6_neighbors(roi_3d, x, y, z)
            density_gradient = self._calculate_density_gradient(roi_3d, x, y, z)
            continuity_score = self._check_continuity_pattern(roi_3d, x, y, z)
            centrality_score = self._calculate_centrality_chebyshev(x, y, z)
            
            # Combine features
            feature_vector = np.concatenate([
                pos_norm,
                [nb_neighbors/6.0],
                [density_gradient],
                [continuity_score],
                [centrality_score]
            ])
            enriched_features.append(feature_vector)
        
        return np.array(enriched_features)
    
    def _count_6_neighbors(self, roi_3d: np.ndarray, x: int, y: int, z: int) -> int:
        """
        Count 6-connected neighbors that are active.
        
        Args:
            roi_3d (np.ndarray): Binary ROI
            x (int): X coordinate
            y (int): Y coordinate  
            z (int): Z coordinate
            
        Returns:
            int: Number of active neighbors (0-6)
        """
        neighbors = [
            (x-1, y, z), (x+1, y, z),
            (x, y-1, z), (x, y+1, z),
            (x, y, z-1), (x, y, z+1)
        ]
        
        count = 0
        for nx, ny, nz in neighbors:
            if (0 <= nx < 30 and 0 <= ny < 38 and 0 <= nz < 22):
                if roi_3d[nx, ny, nz] == 1:
                    count += 1
        
        return count
    
    def _calculate_density_gradient(self, roi_3d: np.ndarray, x: int, y: int, z: int) -> float:
        """
        Calculate local density gradient to detect dense-void transitions.
        
        Args:
            roi_3d (np.ndarray): Binary ROI
            x (int): X coordinate
            y (int): Y coordinate
            z (int): Z coordinate
            
        Returns:
            float: Density gradient score (0-1)
        """
        window = 2
        total_voxels = 0
        active_voxels = 0
        
        for dx in range(-window, window+1):
            for dy in range(-window, window+1):
                for dz in range(-window, window+1):
                    nx, ny, nz = x+dx, y+dy, z+dz
                    if (0 <= nx < 30 and 0 <= ny < 38 and 0 <= nz < 22):
                        total_voxels += 1
                        if roi_3d[nx, ny, nz] == 1:
                            active_voxels += 1
        
        if total_voxels == 0:
            return 0.0
        
        density = active_voxels / total_voxels
        return float(density)
    
    def _check_continuity_pattern(self, roi_3d: np.ndarray, x: int, y: int, z: int) -> float:
        """
        Check for continuity patterns in main directions.
        
        Args:
            roi_3d (np.ndarray): Binary ROI
            x (int): X coordinate
            y (int): Y coordinate
            z (int): Z coordinate
            
        Returns:
            float: Continuity score (0-1)
        """
        directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        continuity_scores = []
        
        for dx, dy, dz in directions:
            # Check continuity in positive direction
            continuous_length = 0
            for step in range(1, 4):
                nx, ny, nz = x + step*dx, y + step*dy, z + step*dz
                if (0 <= nx < 30 and 0 <= ny < 38 and 0 <= nz < 22):
                    if roi_3d[nx, ny, nz] == 1:
                        continuous_length += 1
                    else:
                        break
                else:
                    break
            
            continuity_scores.append(continuous_length / 3.0)
        
        return float(np.mean(continuity_scores))
    
    def _calculate_centrality_chebyshev(self, x: int, y: int, z: int) -> float:
        """
        Calculate centrality score using Chebyshev distance.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            z (int): Z coordinate
            
        Returns:
            float: Centrality score (0-1) where 1=center, 0=border
        """
        center_x, center_y, center_z = 14.5, 18.5, 10.5
        
        dist_x = abs(x - center_x)
        dist_y = abs(y - center_y)
        dist_z = abs(z - center_z)
        
        chebyshev_distance = max(dist_x, dist_y, dist_z)
        max_chebyshev = max(center_x, center_y, center_z)
        
        centrality = 1.0 - (chebyshev_distance / max_chebyshev)
        
        return float(centrality)
    
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
        
        # Apply aggregation (feat_mean_max only)
        feat_mean = x_vis.mean(1)
        feat_max = x_vis.max(1).values
        return torch.cat([feat_mean, feat_max], dim=1)
    
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
            int: Feature dimension (768 for feat_mean_max)
        """
        return 768