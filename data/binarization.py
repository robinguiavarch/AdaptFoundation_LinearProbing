"""
Simple binarization script for HCP OFC skeleton data.
"""

import numpy as np
from pathlib import Path


def binarize_skeleton_file(skeleton_path):
    """
    Load skeleton data, binarize it, and save back to same location.
    """
    skeleton_path = Path(skeleton_path)
    
    
    backup_path = skeleton_path.with_suffix('.npy.backup')
    if not backup_path.exists():
        import shutil
        shutil.copy2(skeleton_path, backup_path)
        print(f"Saving: {backup_path}")
    
    # Load data avec gestion d'erreur
    try:
        data = np.load(skeleton_path)
        print(f"Shape originale: {data.shape}")
        print(f"Valeurs uniques originales: {np.unique(data)}")
        
        # Binarize EN PRÉSERVANT LA FORME
        binary_data = (data != 0).astype(np.uint8)
        
        print(f"Shape après binarisation: {binary_data.shape}")
        print(f"Valeurs uniques après binarisation: {np.unique(binary_data)}")
        
        # Sauvegarder avec la même forme
        np.save(skeleton_path, binary_data)
        print("Binarisation réussie !")
        
    except Exception as e:
        print(f"Erreur lors de la binarisation: {e}")
        # Restaurer depuis la sauvegarde
        if backup_path.exists():
            import shutil
            shutil.copy2(backup_path, skeleton_path)
            print("Fichier restauré depuis la sauvegarde")


def main():
    """
    Main function for standalone execution.
    """
    skeleton_file = "crops/2mm/S.Or./Lskeleton.npy"
    binarize_skeleton_file(skeleton_file)


if __name__ == "__main__":
    main()