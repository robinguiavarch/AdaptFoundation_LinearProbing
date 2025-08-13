"""
Helper pour initialiser SAM-Med3D dans le projet AdaptFoundation
"""
import sys
import os

# Ajouter SAM-Med3D au PYTHONPATH
SAM_MED3D_PATH = '/home/ids/guiavarch-24/SAM-Med3D'
if SAM_MED3D_PATH not in sys.path:
    sys.path.insert(0, SAM_MED3D_PATH)

# Imports SAM-Med3D
from segment_anything.build_sam3D import sam_model_registry3D, build_sam3D_vit_b

print("✅ SAM-Med3D initialisé avec succès")
print(f"   Modèles disponibles: {list(sam_model_registry3D.keys())}")

# Fonction helper pour charger un modèle
def load_sam_med3d_model(model_type='vit_b', checkpoint_path=None):
    """
    Charge un modèle SAM-Med3D
    
    Args:
        model_type: Type de modèle ('default', 'vit_h', 'vit_l', 'vit_b', 'vit_b_ori')
        checkpoint_path: Chemin vers le checkpoint (optionnel)
    
    Returns:
        Modèle SAM-Med3D
    """
    model = sam_model_registry3D[model_type]()
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Checkpoint chargé depuis {checkpoint_path}")
    
    return model
