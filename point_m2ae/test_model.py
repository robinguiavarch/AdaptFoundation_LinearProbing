# test_model.py
import torch
import numpy as np
import sys
import os
sys.path.append('./Point-M2AE')

from models.build import MODELS
from models.Point_M2AE_Finetune import H_Encoder
from data.loaders import HCPOFCDataLoader

class PointM2AEFeatureExtractor:
    """Point-M2AE Encoder seul pour Linear Probing"""
    
    def __init__(self, checkpoint_path):
        # Charger config depuis le checkpoint
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Config Point-M2AE Base (depuis point-m2ae.yaml)
        self.config = self._create_config()
        
        # Initialiser encoder seul (pas decoder)
        self.encoder = H_Encoder(self.config)
        self.group_dividers = self._create_group_dividers()
        
        # Charger poids pré-entraînés
        self._load_checkpoint(checkpoint_path)
        
        self.encoder.eval()
        self.encoder.to(self.device)
    
    def _create_config(self):
        """Config Point-M2AE Base depuis YAML"""
        class Config:
            encoder_depths = [5, 5, 5]
            encoder_dims = [96, 192, 384]
            local_radius = [0.32, 0.64, 1.28]
            drop_path_rate = 0.1
            num_heads = 6
            group_sizes = [16, 8, 8]
            num_groups = [512, 256, 64]
        return Config()
    
    def _create_group_dividers(self):
        """Tokenizers pour groupement points"""
        from models.modules import Group
        group_dividers = torch.nn.ModuleList()
        for i in range(len(self.config.group_sizes)):
            group_dividers.append(Group(
                num_group=self.config.num_groups[i], 
                group_size=self.config.group_sizes[i]
            ))
        return group_dividers
    
    def _load_checkpoint(self, checkpoint_path):
        """Charger poids pré-entraînés"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Le checkpoint contient 'base_model' avec encoder + decoder
        # On extrait seulement les poids encoder
        state_dict = checkpoint['base_model']
        encoder_state_dict = {k: v for k, v in state_dict.items() if 'h_encoder' in k}
        
        # Adapter les clés pour notre H_Encoder
        adapted_state_dict = {}
        for k, v in encoder_state_dict.items():
            new_k = k.replace('h_encoder.', '')  # Retirer préfixe
            adapted_state_dict[new_k] = v
        
        self.encoder.load_state_dict(adapted_state_dict, strict=False)
        print("✅ Checkpoint encoder chargé")
    
    def preprocess_roi(self, roi_volume):
        """ROI (30,38,22) → Point cloud (N,3) normalisé"""
        # Extraire voxels actifs
        active_voxels = np.where(roi_volume == 1)
        if len(active_voxels[0]) == 0:
            raise ValueError("ROI vide, aucun voxel actif")
        
        # Coordonnées points actifs
        points = np.stack(active_voxels, axis=1).astype(np.float32)  # (N, 3)
        
        # Normalisation [-1, 1]
        points = points - points.mean(axis=0)  # Centrage
        points = points / (np.max(np.abs(points)) + 1e-6)  # Scaling
        
        return points
    
    def extract_features(self, roi_volume):
        """ROI → Features globales (384,)"""
        # Preprocessing
        points = self.preprocess_roi(roi_volume)  # (N, 3)
        points_torch = torch.from_numpy(points).unsqueeze(0).to(self.device)  # (1, N, 3)
        
        with torch.no_grad():
            # Multi-scale grouping (comme dans Point_M2AE_Finetune.py)
            neighborhoods, centers, idxs = [], [], []
            for i, group_divider in enumerate(self.group_dividers):
                if i == 0:
                    neighborhood, center, idx = group_divider(points_torch)
                else:
                    neighborhood, center, idx = group_divider(center)
                neighborhoods.append(neighborhood)
                centers.append(center)
                idxs.append(idx)
            
            # Encoder hiérarchique
            x_vis = self.encoder(neighborhoods, centers, idxs, eval=True)
            
            # Agrégation globale (comme ModelNet40/ScanObjectNN)
            global_features = x_vis.mean(1) + x_vis.max(1)[0]  # (1, 384)
            
        return global_features.squeeze(0).cpu().numpy()  # (384,)

def test_feature_extraction():
    """Test extraction 5 ROIs"""
    print("🚀 Test Point-M2AE Feature Extraction")
    
    # 1. Charger dataloader
    try:
        from data.loaders import HCPOFCDataLoader
        loader = HCPOFCDataLoader(
            split='train_val_split_0.csv',  # Premier fold
            batch_size=1
        )
        print(f"✅ DataLoader chargé : {len(loader)} samples")
    except Exception as e:
        print(f"❌ Erreur DataLoader : {e}")
        return
    
    # 2. Initialiser extracteur Point-M2AE
    try:
        extractor = PointM2AEFeatureExtractor('../ckpt/pre-train.pth')
        print("✅ Point-M2AE extractor initialisé")
    except Exception as e:
        print(f"❌ Erreur extracteur : {e}")
        return
    
    # 3. Test extraction sur 5 ROIs
    print("\n📊 Test extraction features :")
    for i, (roi, label, subject_id) in enumerate(loader):
        if i >= 5: break
        
        try:
            roi_np = roi.squeeze(0).numpy()  # (30, 38, 22)
            features = extractor.extract_features(roi_np)
            
            n_active_voxels = np.sum(roi_np == 1)
            print(f"ROI {i+1}: Shape {roi_np.shape}, Voxels actifs: {n_active_voxels}, "
                  f"Features: {features.shape}, Label: {label.item()}")
            
        except Exception as e:
            print(f"❌ Erreur ROI {i+1}: {e}")
    
    print("\n✅ Test extraction terminé !")

if __name__ == "__main__":
    test_feature_extraction()