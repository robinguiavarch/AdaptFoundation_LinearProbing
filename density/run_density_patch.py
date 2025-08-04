#!/usr/bin/env python3
"""
Génération d'une carte de densité par patch 8×8×8 des squelettes sulcaux
pour les sujets non labellisés du dataset HCP OFC.

Cette carte suit exactement le workflow SAM-Med3D :
1. Volumes (30×38×22) → Resize 128³
2. Simulation grille SAM-Med3D 8×8×8
3. Calcul densité moyenne par patch sur 537 sujets non labellisés
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
import time

# Configuration des chemins (depuis la racine du projet)
BASE_PATH = Path("crops/2mm/S.Or.")
SKELETON_FILE = BASE_PATH / "Lskeleton.npy"
SUBJECT_FILE = BASE_PATH / "Lskeleton_subject.csv"
LABELS_FILE = BASE_PATH / "hcp_OFC_labels.csv"
OUTPUT_DIR = Path("density")

# Configuration SAM-Med3D
SAM_INPUT_SIZE = (128, 128, 128)
SAM_PATCH_GRID = (8, 8, 8)  # Grille de patches SAM-Med3D


def create_output_directory(output_dir):
    """Créer le dossier de sortie s'il n'existe pas."""
    output_dir.mkdir(exist_ok=True)
    print(f"Dossier de sortie créé/vérifié : {output_dir}")


def load_data():
    """Charger toutes les données nécessaires."""
    print("Chargement des données...")
    
    # Charger les squelettes
    skeletons = np.load(SKELETON_FILE)
    print(f"Squelettes chargés : {skeletons.shape}")
    
    # Charger le mapping sujets
    subjects_df = pd.read_csv(SUBJECT_FILE)
    print(f"Mapping sujets chargé : {len(subjects_df)} sujets")
    
    # Charger les labels
    labels_df = pd.read_csv(LABELS_FILE)
    print(f"Labels chargés : {len(labels_df)} sujets labellisés")
    
    return skeletons, subjects_df, labels_df


def identify_unlabeled_subjects(subjects_df, labels_df):
    """
    Identifier les sujets non labellisés et leurs indices.
    
    Returns:
        list: Indices des sujets non labellisés dans le tableau de squelettes
    """
    print("\n=== IDENTIFICATION DES SUJETS NON LABELLISÉS ===")
    
    # Ensemble des sujets avec squelettes
    subjects_with_skeletons = set(subjects_df['Subject'].astype(str))
    print(f"Sujets avec squelettes : {len(subjects_with_skeletons)}")
    
    # Ensemble des sujets avec labels
    subjects_with_labels = set(labels_df['Subject'].astype(str))
    print(f"Sujets avec labels : {len(subjects_with_labels)}")
    
    # Sujets non labellisés
    unlabeled_subjects = subjects_with_skeletons - subjects_with_labels
    print(f"Sujets non labellisés : {len(unlabeled_subjects)}")
    
    # Récupérer les indices dans le tableau de squelettes
    unlabeled_indices = []
    for idx, subject in enumerate(subjects_df['Subject']):
        if str(subject) in unlabeled_subjects:
            unlabeled_indices.append(idx)
    
    print(f"Indices récupérés : {len(unlabeled_indices)}")
    
    return unlabeled_indices


def extract_and_resize_volumes(skeletons, unlabeled_indices):
    """
    Extraire les volumes 3D des sujets non labellisés et les resizer vers 128³.
    
    Returns:
        torch.Tensor: Volumes 3D resizés (N, 128, 128, 128)
    """
    print("\n=== EXTRACTION ET RESIZE DES VOLUMES ===")
    
    # Extraire les volumes des sujets non labellisés
    unlabeled_volumes = skeletons[unlabeled_indices]
    print(f"Volumes extraits : {unlabeled_volumes.shape}")
    
    # Supprimer la dimension singleton (dernier axe)
    unlabeled_volumes = unlabeled_volumes.squeeze(axis=-1)
    print(f"Après suppression dimension singleton : {unlabeled_volumes.shape}")
    
    # Convertir en tensor PyTorch pour utiliser F.interpolate
    volumes_tensor = torch.from_numpy(unlabeled_volumes).float()
    print(f"Conversion en tensor : {volumes_tensor.shape}")
    
    # Ajouter dimensions pour F.interpolate [N, C, H, W, D]
    volumes_tensor = volumes_tensor.unsqueeze(1)  # [N, 1, 30, 38, 22]
    print(f"Tensor avec dimension channel : {volumes_tensor.shape}")
    
    # Resize vers 128³ (même méthode que SAM-Med3D)
    print(f"Resize vers {SAM_INPUT_SIZE}...")
    start_time = time.time()
    
    resized_volumes = F.interpolate(
        volumes_tensor,
        size=SAM_INPUT_SIZE,
        mode='trilinear',
        align_corners=False
    )
    
    resize_time = time.time() - start_time
    print(f"Resize terminé en {resize_time:.2f}s")
    print(f"Volumes resizés : {resized_volumes.shape}")
    
    # Retirer la dimension channel pour avoir [N, 128, 128, 128]
    resized_volumes = resized_volumes.squeeze(1)
    print(f"Volumes finaux : {resized_volumes.shape}")
    
    return resized_volumes


def calculate_patch_density_map(resized_volumes):
    """
    Calculer la carte de densité par patch 8×8×8.
    
    Simule la grille de patches SAM-Med3D en divisant 128³ en 8³ patches de 16³.
    
    Returns:
        np.ndarray: Carte de densité par patch (8, 8, 8) avec valeurs entre 0 et 1
    """
    print("\n=== CALCUL DE LA DENSITÉ PAR PATCH ===")
    
    n_subjects, h, w, d = resized_volumes.shape
    print(f"Processing {n_subjects} volumes de taille {h}×{w}×{d}")
    
    # Vérifier que les dimensions sont divisibles par 8
    assert h % 8 == 0 and w % 8 == 0 and d % 8 == 0, \
        f"Les dimensions {h}×{w}×{d} ne sont pas divisibles par 8"
    
    # Taille de chaque patch
    patch_h = h // 8  # 16
    patch_w = w // 8  # 16 
    patch_d = d // 8  # 16
    print(f"Taille des patches : {patch_h}×{patch_w}×{patch_d}")
    
    # Initialiser la carte de densité par patch
    patch_density_sum = np.zeros(SAM_PATCH_GRID, dtype=np.float64)
    
    print("Calcul de la densité pour chaque sujet...")
    start_time = time.time()
    
    for subject_idx in range(n_subjects):
        volume = resized_volumes[subject_idx].numpy()
        
        # Calculer la densité pour chaque patch 8×8×8
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    # Extraire le patch 16³
                    patch = volume[
                        i*patch_h:(i+1)*patch_h,
                        j*patch_w:(j+1)*patch_w,
                        k*patch_d:(k+1)*patch_d
                    ]
                    
                    # Calculer la densité du patch (proportion de voxels > 0)
                    patch_density = np.mean(patch > 0)
                    patch_density_sum[i, j, k] += patch_density
        
        # Affichage du progrès
        if (subject_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {subject_idx + 1}/{n_subjects} sujets ({elapsed:.1f}s)")
    
    # Calculer la densité moyenne par patch
    patch_density_map = patch_density_sum / n_subjects
    
    calc_time = time.time() - start_time
    print(f"Calcul terminé en {calc_time:.2f}s")
    print(f"Forme de la carte de densité par patch : {patch_density_map.shape}")
    print(f"Valeurs min/max : {np.min(patch_density_map):.4f} / {np.max(patch_density_map):.4f}")
    print(f"Valeur moyenne : {np.mean(patch_density_map):.4f}")
    print(f"Nombre de patches avec densité = 0 : {np.sum(patch_density_map == 0)}")
    print(f"Nombre de patches avec densité > 0 : {np.sum(patch_density_map > 0)}")
    
    # Vérification que les valeurs sont bien dans [0,1]
    assert np.all(patch_density_map >= 0) and np.all(patch_density_map <= 1), \
        "Erreur : valeurs hors de [0,1]"
    print("✓ Validation : toutes les valeurs sont dans [0,1]")
    
    return patch_density_map


def save_patch_density_map(patch_density_map, output_dir):
    """Sauvegarder la carte de densité par patch."""
    output_file = output_dir / "patch_density_map_8x8x8.npy"
    np.save(output_file, patch_density_map)
    print(f"\n✓ Carte de densité par patch sauvegardée : {output_file}")
    print(f"Taille du fichier : {output_file.stat().st_size / 1024:.1f} KB")
    
    # Sauvegarder aussi en format lisible pour inspection
    stats_file = output_dir / "patch_density_stats.txt"
    with open(stats_file, 'w') as f:
        f.write("STATISTIQUES CARTE DE DENSITÉ PAR PATCH 8×8×8\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Forme : {patch_density_map.shape}\n")
        f.write(f"Valeur min : {np.min(patch_density_map):.6f}\n")
        f.write(f"Valeur max : {np.max(patch_density_map):.6f}\n")
        f.write(f"Valeur moyenne : {np.mean(patch_density_map):.6f}\n")
        f.write(f"Écart-type : {np.std(patch_density_map):.6f}\n")
        f.write(f"Patches avec densité = 0 : {np.sum(patch_density_map == 0)}\n")
        f.write(f"Patches avec densité > 0 : {np.sum(patch_density_map > 0)}\n")
        f.write(f"Patches avec densité > 0.1 : {np.sum(patch_density_map > 0.1)}\n")
        f.write(f"Patches avec densité > 0.5 : {np.sum(patch_density_map > 0.5)}\n")
        
        # Distribution par tranche
        f.write("\nDISTRIBUTION PAR TRANCHE :\n")
        for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            count = np.sum(patch_density_map > threshold)
            percentage = count / patch_density_map.size * 100
            f.write(f"  > {threshold:.1f} : {count:3d} patches ({percentage:5.1f}%)\n")
    
    print(f"✓ Statistiques sauvegardées : {stats_file}")


def main():
    """Fonction principale."""
    print("=" * 70)
    print("GÉNÉRATION DE LA CARTE DE DENSITÉ PAR PATCH 8×8×8")
    print("Workflow SAM-Med3D : (30×38×22) → 128³ → patches 8×8×8")
    print("=" * 70)
    
    # Créer le dossier de sortie
    create_output_directory(OUTPUT_DIR)
    
    # Charger les données
    skeletons, subjects_df, labels_df = load_data()
    
    # Identifier les sujets non labellisés
    unlabeled_indices = identify_unlabeled_subjects(subjects_df, labels_df)
    
    # Extraire et resizer les volumes
    resized_volumes = extract_and_resize_volumes(skeletons, unlabeled_indices)
    
    # Calculer la carte de densité par patch
    patch_density_map = calculate_patch_density_map(resized_volumes)
    
    # Sauvegarder
    save_patch_density_map(patch_density_map, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("GÉNÉRATION TERMINÉE AVEC SUCCÈS !")
    print("=" * 70)
    print("Fichiers générés :")
    print(f"  - density/patch_density_map_8x8x8.npy")
    print(f"  - density/patch_density_stats.txt")
    print("\nProchaine étape : Utiliser cette carte dans feature_extraction_core.py")


if __name__ == "__main__":
    main()