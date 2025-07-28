"""
Calcul des profils de densité 1D selon les axes X, Y et Z
à partir de la carte de densité volumique 3D.
"""

import numpy as np
from pathlib import Path

# Configuration des chemins (depuis la racine du projet)
DENSITY_MAP_FILE = "density/density_map_unlabeled_subjects.npy"
OUTPUT_DIR = Path("density")

def load_density_map(file_path):
    """Charger la carte de densité 3D."""
    density_map = np.load(file_path)
    print(f"Carte de densité chargée : {density_map.shape}")
    print(f"Valeurs min/max : {np.min(density_map):.6f} / {np.max(density_map):.6f}")
    return density_map

def calculate_density_profile_x(density_map):
    """
    Calculer le profil de densité selon l'axe X.
    
    Args:
        density_map (np.ndarray): Carte 3D de forme (30, 38, 22)
        
    Returns:
        np.ndarray: Profil 1D de forme (30,)
    """
    print("\n=== CALCUL DU PROFIL SELON X ===")
    
    num_slices_x = density_map.shape[0]  # 30
    profile_x = np.zeros(num_slices_x)
    
    for i in range(num_slices_x):
        # Prendre la slice selon X : density_map[i, :, :]
        slice_x = density_map[i, :, :]  # Shape (38, 22)
        
        # Sommer toutes les valeurs de la slice
        total_sum = np.sum(slice_x)
        
        # Diviser par le nombre total de voxels de la slice
        num_voxels = slice_x.shape[0] * slice_x.shape[1]  # 38 * 22 = 836
        
        # Densité moyenne de cette slice
        profile_x[i] = total_sum / num_voxels
    
    print(f"Profil X calculé : {profile_x.shape}")
    print(f"Valeurs min/max : {np.min(profile_x):.6f} / {np.max(profile_x):.6f}")
    print(f"Valeur moyenne : {np.mean(profile_x):.6f}")
    
    return profile_x

def calculate_density_profile_y(density_map):
    """
    Calculer le profil de densité selon l'axe Y.
    
    Args:
        density_map (np.ndarray): Carte 3D de forme (30, 38, 22)
        
    Returns:
        np.ndarray: Profil 1D de forme (38,)
    """
    print("\n=== CALCUL DU PROFIL SELON Y ===")
    
    num_slices_y = density_map.shape[1]  # 38
    profile_y = np.zeros(num_slices_y)
    
    for j in range(num_slices_y):
        # Prendre la slice selon Y : density_map[:, j, :]
        slice_y = density_map[:, j, :]  # Shape (30, 22)
        
        # Sommer toutes les valeurs de la slice
        total_sum = np.sum(slice_y)
        
        # Diviser par le nombre total de voxels de la slice
        num_voxels = slice_y.shape[0] * slice_y.shape[1]  # 30 * 22 = 660
        
        # Densité moyenne de cette slice
        profile_y[j] = total_sum / num_voxels
    
    print(f"Profil Y calculé : {profile_y.shape}")
    print(f"Valeurs min/max : {np.min(profile_y):.6f} / {np.max(profile_y):.6f}")
    print(f"Valeur moyenne : {np.mean(profile_y):.6f}")
    
    return profile_y

def calculate_density_profile_z(density_map):
    """
    Calculer le profil de densité selon l'axe Z.
    
    Args:
        density_map (np.ndarray): Carte 3D de forme (30, 38, 22)
        
    Returns:
        np.ndarray: Profil 1D de forme (22,)
    """
    print("\n=== CALCUL DU PROFIL SELON Z ===")
    
    num_slices_z = density_map.shape[2]  # 22
    profile_z = np.zeros(num_slices_z)
    
    for k in range(num_slices_z):
        # Prendre la slice selon Z : density_map[:, :, k]
        slice_z = density_map[:, :, k]  # Shape (30, 38)
        
        # Sommer toutes les valeurs de la slice
        total_sum = np.sum(slice_z)
        
        # Diviser par le nombre total de voxels de la slice
        num_voxels = slice_z.shape[0] * slice_z.shape[1]  # 30 * 38 = 1140
        
        # Densité moyenne de cette slice
        profile_z[k] = total_sum / num_voxels
    
    print(f"Profil Z calculé : {profile_z.shape}")
    print(f"Valeurs min/max : {np.min(profile_z):.6f} / {np.max(profile_z):.6f}")
    print(f"Valeur moyenne : {np.mean(profile_z):.6f}")
    
    return profile_z

def save_profiles(profile_x, profile_y, profile_z, output_dir):
    """Sauvegarder les trois profils de densité."""
    
    # Fichiers de sortie
    file_x = output_dir / "density_profile_x.npy"
    file_y = output_dir / "density_profile_y.npy"
    file_z = output_dir / "density_profile_z.npy"
    
    # Sauvegarde
    np.save(file_x, profile_x)
    np.save(file_y, profile_y)
    np.save(file_z, profile_z)
    
    print(f"\n=== SAUVEGARDE TERMINÉE ===")
    print(f"✓ Profil X sauvegardé : {file_x}")
    print(f"✓ Profil Y sauvegardé : {file_y}")
    print(f"✓ Profil Z sauvegardé : {file_z}")
    
    # Informations sur les fichiers
    for file_path, shape in [(file_x, profile_x.shape), (file_y, profile_y.shape), (file_z, profile_z.shape)]:
        size_kb = file_path.stat().st_size / 1024
        print(f"  {file_path.name}: {shape} - {size_kb:.1f} KB")

def verify_consistency(density_map, profile_x, profile_y, profile_z):
    """Vérifier la cohérence des calculs."""
    print(f"\n=== VÉRIFICATION DE COHÉRENCE ===")
    
    # La moyenne de tous les profils devrait être proche de la moyenne globale
    global_mean = np.mean(density_map)
    
    # Moyenne pondérée des profils (pondérée par le nombre de voxels par slice)
    mean_x = np.mean(profile_x)  # Chaque slice X a 38*22 voxels
    mean_y = np.mean(profile_y)  # Chaque slice Y a 30*22 voxels  
    mean_z = np.mean(profile_z)  # Chaque slice Z a 30*38 voxels
    
    print(f"Moyenne globale de la carte 3D : {global_mean:.6f}")
    print(f"Moyenne du profil X : {mean_x:.6f}")
    print(f"Moyenne du profil Y : {mean_y:.6f}")
    print(f"Moyenne du profil Z : {mean_z:.6f}")
    
    # Ces valeurs devraient être identiques
    tolerance = 1e-10
    x_ok = abs(global_mean - mean_x) < tolerance
    y_ok = abs(global_mean - mean_y) < tolerance
    z_ok = abs(global_mean - mean_z) < tolerance
    
    print(f"Cohérence X : {'✓' if x_ok else '✗'}")
    print(f"Cohérence Y : {'✓' if y_ok else '✗'}")
    print(f"Cohérence Z : {'✓' if z_ok else '✗'}")
    
    if x_ok and y_ok and z_ok:
        print("✓ Tous les calculs sont cohérents !")
    else:
        print("⚠ Incohérence détectée dans les calculs")

def main():
    """Fonction principale."""
    print("=" * 60)
    print("CALCUL DES PROFILS DE DENSITÉ 1D")
    print("=" * 60)
    
    # Charger la carte de densité 3D
    density_map = load_density_map(DENSITY_MAP_FILE)
    
    # Calculer les trois profils
    profile_x = calculate_density_profile_x(density_map)
    profile_y = calculate_density_profile_y(density_map)
    profile_z = calculate_density_profile_z(density_map)
    
    # Vérifier la cohérence
    verify_consistency(density_map, profile_x, profile_y, profile_z)
    
    # Sauvegarder les profils
    save_profiles(profile_x, profile_y, profile_z, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("CALCUL DES PROFILS TERMINÉ AVEC SUCCÈS !")
    print("=" * 60)

if __name__ == "__main__":
    main()