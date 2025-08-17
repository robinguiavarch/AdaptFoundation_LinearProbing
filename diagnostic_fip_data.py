#!/usr/bin/env python3
"""
Script de diagnostic pour identifier le problème des résultats identiques
entre PCA_256, PCA_95 et PCA_995.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def diagnostic_fip_data():
    """Diagnostic complet des données F.I.P."""
    
    base_path = Path("feature_extraction_sam3d_fip/flatten")
    
    print("=== DIAGNOSTIC F.I.P. DATA ===")
    
    # 1. Vérifier les shapes
    print("\n1. VERIFICATION DES SHAPES:")
    for pca_mode in ["32", "256", "95", "995"]:
        pca_dir = base_path / f"PCA_{pca_mode}"
        if pca_dir.exists():
            features_file = pca_dir / "train_val_split_0_features.npy"
            if features_file.exists():
                features = np.load(features_file)
                print(f"  PCA_{pca_mode}: {features.shape}")
            else:
                print(f"  PCA_{pca_mode}: FICHIER MANQUANT")
        else:
            print(f"  PCA_{pca_mode}: REPERTOIRE MANQUANT")
    
    # 2. Vérifier si les données sont identiques
    print("\n2. VERIFICATION DES DONNEES:")
    
    # Charger les données
    data = {}
    for pca_mode in ["32", "256", "95", "995"]:
        pca_dir = base_path / f"PCA_{pca_mode}"
        features_file = pca_dir / "train_val_split_0_features.npy"
        if features_file.exists():
            data[pca_mode] = np.load(features_file)
    
    # Comparer les premières colonnes
    if "256" in data and "95" in data:
        print(f"  PCA_256 vs PCA_95 (premières 256 cols): {np.array_equal(data['256'], data['95'][:, :256])}")
        print(f"  PCA_256 vs PCA_95 (premières 10 valeurs): {np.allclose(data['256'][:5, :10], data['95'][:5, :10])}")
    
    if "256" in data and "995" in data:
        print(f"  PCA_256 vs PCA_995 (premières 256 cols): {np.array_equal(data['256'], data['995'][:, :256])}")
        print(f"  PCA_256 vs PCA_995 (premières 10 valeurs): {np.allclose(data['256'][:5, :10], data['995'][:5, :10])}")
    
    if "95" in data and "995" in data:
        print(f"  PCA_95 vs PCA_995 (premières 273 cols): {np.array_equal(data['95'], data['995'][:, :273])}")
        print(f"  PCA_95 vs PCA_995 (premières 10 valeurs): {np.allclose(data['95'][:5, :10], data['995'][:5, :10])}")
    
    # 3. Vérifier les métadonnées
    print("\n3. VERIFICATION DES METADONNEES:")
    for pca_mode in ["32", "256", "95", "995"]:
        pca_dir = base_path / f"PCA_{pca_mode}"
        metadata_file = pca_dir / "train_val_split_0_metadata.csv"
        if metadata_file.exists():
            metadata = pd.read_csv(metadata_file)
            labels = metadata['Label'].values
            subjects = metadata['Subject'].values
            print(f"  PCA_{pca_mode}: {len(labels)} samples, labels={np.unique(labels)}, premiers sujets={subjects[:3]}")
    
    # 4. Vérifier si les labels sont identiques
    print("\n4. VERIFICATION DES LABELS:")
    labels_data = {}
    for pca_mode in ["32", "256", "95", "995"]:
        pca_dir = base_path / f"PCA_{pca_mode}"
        metadata_file = pca_dir / "train_val_split_0_metadata.csv"
        if metadata_file.exists():
            metadata = pd.read_csv(metadata_file)
            labels_data[pca_mode] = metadata['Label'].values
    
    for pca1 in ["256", "95", "995"]:
        for pca2 in ["256", "95", "995"]:
            if pca1 < pca2 and pca1 in labels_data and pca2 in labels_data:
                identical = np.array_equal(labels_data[pca1], labels_data[pca2])
                print(f"  Labels PCA_{pca1} vs PCA_{pca2}: {identical}")
    
    # 5. Vérifier les statistiques des features
    print("\n5. STATISTIQUES DES FEATURES:")
    for pca_mode in ["32", "256", "95", "995"]:
        if pca_mode in data:
            features = data[pca_mode]
            print(f"  PCA_{pca_mode}:")
            print(f"    Mean: {features.mean():.6f}")
            print(f"    Std: {features.std():.6f}")
            print(f"    Min: {features.min():.6f}")
            print(f"    Max: {features.max():.6f}")
            print(f"    Première valeur: {features[0, 0]:.6f}")
    
    # 6. Hashing pour vérifier l'identité
    print("\n6. HASH DES DONNEES:")
    for pca_mode in ["32", "256", "95", "995"]:
        if pca_mode in data:
            data_hash = hash(data[pca_mode].tobytes())
            print(f"  PCA_{pca_mode} hash: {data_hash}")
    
    print("\n=== FIN DIAGNOSTIC ===")

if __name__ == "__main__":
    diagnostic_fip_data()