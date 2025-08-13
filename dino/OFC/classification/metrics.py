"""
Classification metrics module for AdaptFoundation project.

This module implements specialized metrics for multi-class classification
evaluation, particularly ROC-AUC calculations for foundation model assessment.
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer


class ClassificationMetrics:
    """
    Comprehensive classification metrics calculator for multi-class problems.
    
    Provides ROC-AUC calculations, confusion matrices, and detailed performance
    reports for foundation model evaluation.
    
    Attributes:
        n_classes (int): Number of classes in the classification problem
        class_names (List[str]): Names of the classes
    """
    
    def __init__(self, n_classes: int = 4, 
                 class_names: List[str] = None):
        """
        Initialize the metrics calculator.
        
        Args:
            n_classes (int): Number of classes. Defaults to 4.
            class_names (List[str], optional): Class names. Defaults to None.
        """
        self.n_classes = n_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(n_classes)]
    
    def calculate_roc_auc_ovr(self, y_true: np.ndarray, 
                             y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate ROC-AUC One-vs-Rest for multi-class classification.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
        
        Returns:
            Dict[str, float]: ROC-AUC scores (per-class and weighted average)
        """
        # Overall weighted ROC-AUC
        roc_auc_weighted = roc_auc_score(y_true, y_pred_proba, 
                                       multi_class='ovr', average='weighted')
        
        # Macro average ROC-AUC
        roc_auc_macro = roc_auc_score(y_true, y_pred_proba, 
                                    multi_class='ovr', average='macro')
        
        # Per-class ROC-AUC
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(y_true)
        
        per_class_auc = {}
        for i, class_name in enumerate(self.class_names):
            if y_bin.shape[1] > 1:  # Multi-class case
                class_auc = roc_auc_score(y_bin[:, i], y_pred_proba[:, i])
            else:  # Binary case
                class_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            per_class_auc[class_name] = class_auc
        
        return {
            'roc_auc_weighted': roc_auc_weighted,
            'roc_auc_macro': roc_auc_macro,
            'per_class_auc': per_class_auc
        }
    
    def calculate_roc_auc_for_predictions(self, y_true: np.ndarray, 
                                        y_pred: np.ndarray) -> float:
        """
        Calculate ROC-AUC for models without probability prediction.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        
        Returns:
            float: ROC-AUC weighted score
        """
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_true)
        y_pred_bin = lb.transform(y_pred)
        
        return roc_auc_score(y_true_bin, y_pred_bin, average='weighted')
    
    def generate_classification_report(self, y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> Dict:
        """
        Generate comprehensive classification report.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        
        Returns:
            Dict: Classification report with precision, recall, F1-score
        """
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names,
                                     output_dict=True, zero_division=0)
        
        return report
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        
        Returns:
            np.ndarray: Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_pred_proba: np.ndarray = None) -> Dict:
        """
        Complete evaluation of predictions with all metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray, optional): Predicted probabilities
        
        Returns:
            Dict: Complete evaluation results
        """
        results = {}
        
        # Classification report
        results['classification_report'] = self.generate_classification_report(y_true, y_pred)
        
        # Confusion matrix
        results['confusion_matrix'] = self.calculate_confusion_matrix(y_true, y_pred)
        
        # ROC-AUC calculations
        if y_pred_proba is not None:
            results['roc_auc'] = self.calculate_roc_auc_ovr(y_true, y_pred_proba)
        else:
            results['roc_auc'] = {
                'roc_auc_weighted': self.calculate_roc_auc_for_predictions(y_true, y_pred)
            }
        
        # Overall accuracy
        results['accuracy'] = np.mean(y_true == y_pred)
        
        return results