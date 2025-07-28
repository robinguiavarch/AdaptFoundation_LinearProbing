"""
Génération d'une carte de densité volumique 3D des squelettes sulcaux
pour les sujets non labellisés du dataset HCP OFC.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Configuration des chemins (depuis la racine du projet)
BASE_PATH = Path("crops/2mm/S.Or.")
SKELETON_FILE = BASE_PATH / "Lskeleton.npy"
SUBJECT_FILE = BASE_PATH / "Lskeleton_subject.csv"
LABELS_FILE = BASE_PATH / "hcp_OFC_labels.csv"
OUTPUT_DIR = Path("density")

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
    Étape 1 : Identifier les sujets non labellisés et leurs indices.
    
    Returns:
        list: Indices des sujets non labellisés dans le tableau de squelettes
    """
    print("\n=== ÉTAPE 1 : Identification des sujets non labellisés ===")
    
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
    print(f"Vérification : {len(unlabeled_indices)} == {len(unlabeled_subjects)}")
    
    return unlabeled_indices

def extract_unlabeled_volumes(skeletons, unlabeled_indices):
    """
    Étape 2 : Extraire les volumes 3D des sujets non labellisés.
    
    Returns:
        np.ndarray: Volumes 3D des sujets non labellisés (N, 30, 38, 22)
    """
    print("\n=== ÉTAPE 2 : Extraction des volumes 3D ===")
    
    # Extraire les volumes des sujets non labellisés
    unlabeled_volumes = skeletons[unlabeled_indices]
    print(f"Volumes extraits : {unlabeled_volumes.shape}")
    
    # Supprimer la dimension singleton (dernier axe)
    unlabeled_volumes = unlabeled_volumes.squeeze(axis=-1)
    print(f"Après suppression dimension singleton : {unlabeled_volumes.shape}")
    
    # Vérification du type de données
    print(f"Type de données : {unlabeled_volumes.dtype}")
    print(f"Valeurs uniques : {np.unique(unlabeled_volumes)}")
    
    return unlabeled_volumes

def calculate_density_map(unlabeled_volumes):
    """
    Étape 3 : Calculer la carte de densité volumique.
    
    Returns:
        np.ndarray: Carte de densité (30, 38, 22) avec valeurs entre 0 et 1
    """
    print("\n=== ÉTAPE 3 : Calcul de la carte de densité ===")
    
    # Sommer tous les volumes (addition voxel par voxel)
    total_sum = np.sum(unlabeled_volumes, axis=0)
    print(f"Forme de la somme : {total_sum.shape}")
    print(f"Valeur max de la somme : {np.max(total_sum)}")
    
    # Diviser par le nombre de sujets pour obtenir la fréquence moyenne
    num_subjects = unlabeled_volumes.shape[0]
    density_map = total_sum / num_subjects
    
    print(f"Nombre de sujets utilisés : {num_subjects}")
    print(f"Forme de la carte de densité : {density_map.shape}")
    print(f"Valeurs min/max : {np.min(density_map):.4f} / {np.max(density_map):.4f}")
    print(f"Valeur moyenne : {np.mean(density_map):.4f}")
    
    # Vérification que les valeurs sont bien dans [0,1]
    assert np.all(density_map >= 0) and np.all(density_map <= 1), "Erreur : valeurs hors de [0,1]"
    print("✓ Validation : toutes les valeurs sont dans [0,1]")
    
    return density_map

def save_density_map(density_map, output_dir):
    """Sauvegarder la carte de densité."""
    output_file = output_dir / "density_map_unlabeled_subjects.npy"
    np.save(output_file, density_map)
    print(f"\n✓ Carte de densité sauvegardée : {output_file}")
    print(f"Taille du fichier : {output_file.stat().st_size / 1024:.1f} KB")

def main():
    """Fonction principale."""
    print("=" * 60)
    print("GÉNÉRATION DE LA CARTE DE DENSITÉ VOLUMIQUE 3D")
    print("=" * 60)
    
    # Créer le dossier de sortie
    create_output_directory(OUTPUT_DIR)
    
    # Charger les données
    skeletons, subjects_df, labels_df = load_data()
    
    # Étape 1 : Identifier les sujets non labellisés
    unlabeled_indices = identify_unlabeled_subjects(subjects_df, labels_df)
    
    # Étape 2 : Extraire les volumes 3D
    unlabeled_volumes = extract_unlabeled_volumes(skeletons, unlabeled_indices)
    
    # Étape 3 : Calculer la carte de densité
    density_map = calculate_density_map(unlabeled_volumes)
    
    # Sauvegarder
    save_density_map(density_map, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("GÉNÉRATION TERMINÉE AVEC SUCCÈS !")
    print("=" * 60)

if __name__ == "__main__":
    main()