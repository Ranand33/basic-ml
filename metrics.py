import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Dict, Tuple, Callable
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

class EvaluationMetrics:
    """A comprehensive collection of evaluation metrics for classification and regression models."""
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float: 
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, normalize: Optional[str] = None) -> np.ndarray:
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize == 'true':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        elif normalize == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        elif normalize == 'all':
            cm = cm.astype('float') / cm.sum()
            
        return cm
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary', 
                  pos_label: int = 1, labels: Optional[List] = None) -> Union[float, np.ndarray]:
       
        if average == 'binary':
            cm = confusion_matrix(y_true, y_pred)
            true_pos = cm[pos_label, pos_label]
            false_pos = np.sum(cm[:, pos_label]) - true_pos
            return true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        
        # Multi-class case
        precision_scores = []
        unique_labels = labels if labels is not None else np.unique(np.concatenate([y_true, y_pred]))
        
        for label in unique_labels:
            y_true_binary = (y_true == label).astype(int)
            y_pred_binary = (y_pred == label).astype(int)
            cm = confusion_matrix(y_true_binary, y_pred_binary)
            true_pos = cm[1, 1]
            false_pos = cm[0, 1]
            precision_scores.append(true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0)
        
        if average == 'micro':
            # Recompute using all TPs and FPs
            true_pos_sum = 0
            false_pos_sum = 0
            for label in unique_labels:
                y_true_binary = (y_true == label).astype(int)
                y_pred_binary = (y_pred == label).astype(int)
                cm = confusion_matrix(y_true_binary, y_pred_binary)
                true_pos_sum += cm[1, 1]
                false_pos_sum += cm[0, 1]
            return true_pos_sum / (true_pos_sum + false_pos_sum) if (true_pos_sum + false_pos_sum) > 0 else 0.0
            
        elif average == 'macro':
            return np.mean(precision_scores)
            
        elif average == 'weighted':
            weights = np.array([np.sum(y_true == label) for label in unique_labels])
            return np.average(precision_scores, weights=weights) if np.sum(weights) > 0 else 0.0
            
        elif average is None:
            return np.array(precision_scores)
        
        else:
            raise ValueError(f"Unsupported average type: {average}")
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary', 
              pos_label: int = 1, labels: Optional[List] = None) -> Union[float, np.ndarray]:
        
        if average == 'binary':
            cm = confusion_matrix(y_true, y_pred)
            true_pos = cm[pos_label, pos_label]
            false_neg = np.sum(cm[pos_label, :]) - true_pos
            return true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        
        # Multi-class case
        recall_scores = []
        unique_labels = labels if labels is not None else np.unique(np.concatenate([y_true, y_pred]))
        
        for label in unique_labels:
            y_true_binary = (y_true == label).astype(int)
            y_pred_binary = (y_pred == label).astype(int)
            cm = confusion_matrix(y_true_binary, y_pred_binary)
            true_pos = cm[1, 1]
            false_neg = cm[1, 0]
            recall_scores.append(true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0)
        
        if average == 'micro':
            # Recompute using all TPs and FNs
            true_pos_sum = 0
            false_neg_sum = 0
            for label in unique_labels:
                y_true_binary = (y_true == label).astype(int)
                y_pred_binary = (y_pred == label).astype(int)
                cm = confusion_matrix(y_true_binary, y_pred_binary)
                true_pos_sum += cm[1, 1]
                false_neg_sum += cm[1, 0]
            return true_pos_sum / (true_pos_sum + false_neg_sum) if (true_pos_sum + false_neg_sum) > 0 else 0.0
            
        elif average == 'macro':
            return np.mean(recall_scores)
            
        elif average == 'weighted':
            weights = np.array([np.sum(y_true == label) for label in unique_labels])
            return np.average(recall_scores, weights=weights) if np.sum(weights) > 0 else 0.0
            
        elif average is None:
            return np.array(recall_scores)
        
        else:
            raise ValueError(f"Unsupported average type: {average}")
    
    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary', 
                pos_label: int = 1, labels: Optional[List] = None) -> Union[float, np.ndarray]:
        
        precision = EvaluationMetrics.precision(y_true, y_pred, average, pos_label, labels)
        recall = EvaluationMetrics.recall(y_true, y_pred, average, pos_label, labels)
        
        if average is None:
            return 2 * precision * recall / (precision + recall + 1e-10)
        else:
            return 2 * precision * recall / (precision + recall + 1e-10)
    
    @staticmethod
    def roc_auc(y_true: np.ndarray, y_score: np.ndarray, multi_class: str = 'ovr') -> float:
      
        # Ensure binary case is correctly formatted
        unique_classes = np.unique(y_true)
        
        if len(unique_classes) <= 2:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            return np.trapz(tpr, fpr)  # Approximate AUC using the trapezoidal rule
        
        # Multi-class case
        if multi_class == 'ovr':
            # One-vs-rest approach
            auc_scores = []
            
            # Convert y_score to a matrix if it's not already
            if y_score.ndim == 1:
                raise ValueError("For multi-class ROC AUC, y_score should be a 2D array with shape (n_samples, n_classes)")
            
            for i, cls in enumerate(unique_classes):
                cls_true = (y_true == cls).astype(int)
                cls_score = y_score[:, i]
                fpr, tpr, _ = roc_curve(cls_true, cls_score)
                auc_scores.append(np.trapz(tpr, fpr))
            
            return np.mean(auc_scores)
        
        elif multi_class == 'ovo':
            # One-vs-one approach
            n_classes = len(unique_classes)
            auc_scores = []
            
            for i in range(n_classes):
                for j in range(i+1, n_classes):
                    mask = np.logical_or(y_true == unique_classes[i], y_true == unique_classes[j])
                    binary_true = (y_true[mask] == unique_classes[i]).astype(int)
                    binary_score = y_score[mask, i] - y_score[mask, j]
                    
                    fpr, tpr, _ = roc_curve(binary_true, binary_score)
                    auc_scores.append(np.trapz(tpr, fpr))
            
            return np.mean(auc_scores)
        
        else:
            raise ValueError(f"Unsupported multi_class type: {multi_class}")
    
    @staticmethod
    def precision_recall_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return np.trapz(precision, recall)
    
    @staticmethod
    def specificity(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
       
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape[0] <= 2:
            if pos_label == 1:
                tn = cm[0, 0]
                fp = cm[0, 1]
            else:
                tn = cm[1, 1]
                fp = cm[1, 0]
        else:
            # Multi-class: treat specified label as positive, all others as negative
            neg_indices = [i for i in range(cm.shape[0]) if i != pos_label]
            tn = np.sum(cm[neg_indices, :][:, neg_indices])
            fp = np.sum(cm[neg_indices, pos_label])
        
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    @staticmethod
    def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape[0] == 2:
            tn, fp, fn, tp = cm.ravel()
            
            numerator = tp * tn - fp * fn
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            
            return numerator / denominator if denominator != 0 else 0.0
        
        else:
            # Multi-class MCC
            n_classes = cm.shape[0]
            n_samples = np.sum(cm)
            
            # Calculate cov(X, Y)
            c_xy = 0
            for i in range(n_classes):
                for j in range(n_classes):
                    c_xy += cm[i, j] * (n_samples * int(i == j) - np.sum(cm[i, :]) * np.sum(cm[:, j]))
            
            # Calculate cov(X, X) and cov(Y, Y)
            c_xx = n_samples**2 - np.sum([np.sum(cm[i, :])**2 for i in range(n_classes)])
            c_yy = n_samples**2 - np.sum([np.sum(cm[:, j])**2 for j in range(n_classes)])
            
            return c_xy / np.sqrt(c_xx * c_yy) if (c_xx * c_yy) > 0 else 0.0
    
    @staticmethod
    def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[str] = None) -> float:
        
        cm = confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]
        n_samples = np.sum(cm)
        
        p_o = np.sum(np.diag(cm)) / n_samples  # Observed agreement
        
        # Expected agreement
        p_e = 0
        if weights is None:
            # Unweighted kappa
            for i in range(n_classes):
                p_e += (np.sum(cm[i, :]) / n_samples) * (np.sum(cm[:, i]) / n_samples)
                
        elif weights == 'linear':
            # Linear weighted kappa
            for i in range(n_classes):
                for j in range(n_classes):
                    p_e += (np.sum(cm[i, :]) / n_samples) * (np.sum(cm[:, j]) / n_samples) * (1 - abs(i - j) / (n_classes - 1))
                    
        elif weights == 'quadratic':
            # Quadratic weighted kappa
            for i in range(n_classes):
                for j in range(n_classes):
                    p_e += (np.sum(cm[i, :]) / n_samples) * (np.sum(cm[:, j]) / n_samples) * (1 - ((i - j) / (n_classes - 1))**2)
        
        else:
            raise ValueError(f"Unsupported weights type: {weights}")
        
        return (p_o - p_e) / (1 - p_e) if p_e != 1 else 0.0
    
    @staticmethod
    def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
       
        unique_classes = np.unique(y_true)
        class_accuracies = []
        
        for cls in unique_classes:
            mask = (y_true == cls)
            class_accuracies.append(np.mean(y_pred[mask] == cls))
        
        return np.mean(class_accuracies)
    
    # Regression Metrics
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, 
                           sample_weight: Optional[np.ndarray] = None) -> float:
       
        errors = np.abs(y_true - y_pred)
        
        if sample_weight is not None:
            return np.average(errors, weights=sample_weight)
        else:
            return np.mean(errors)
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, 
                          sample_weight: Optional[np.ndarray] = None,
                          squared: bool = True) -> float:
      
        errors = (y_true - y_pred) ** 2
        
        if sample_weight is not None:
            mse = np.average(errors, weights=sample_weight)
        else:
            mse = np.mean(errors)
        
        return mse if squared else np.sqrt(mse)
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, 
                               sample_weight: Optional[np.ndarray] = None) -> float:
        return EvaluationMetrics.mean_squared_error(y_true, y_pred, sample_weight, squared=False)
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, 
                                      sample_weight: Optional[np.ndarray] = None) -> float:
       
        # Avoid division by zero
        mask = y_true != 0
        y_true_safe = y_true[mask]
        y_pred_safe = y_pred[mask]
        
        if len(y_true_safe) == 0:
            return np.nan
        
        errors = np.abs((y_true_safe - y_pred_safe) / np.abs(y_true_safe)) * 100
        
        if sample_weight is not None:
            sample_weight_safe = sample_weight[mask]
            return np.average(errors, weights=sample_weight_safe)
        else:
            return np.mean(errors)
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray, 
                sample_weight: Optional[np.ndarray] = None,
                multioutput: str = 'uniform_average') -> Union[float, np.ndarray]:
      
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            weight = sample_weight[:, np.newaxis]
        else:
            weight = 1
        
        # Calculate weighted mean of y_true
        y_true_mean = np.average(y_true, axis=0, weights=sample_weight)
        
        # Numerator: weighted sum of squared residuals
        numerator = np.sum(weight * (y_true - y_pred) ** 2, axis=0)
        
        # Denominator: weighted total sum of squares
        denominator = np.sum(weight * (y_true - y_true_mean) ** 2, axis=0)
        
        # Avoid division by zero
        nonzero_denominator = denominator != 0
        
        r2 = np.ones(y_true.shape[1])
        r2[nonzero_denominator] = 1 - (numerator[nonzero_denominator] / denominator[nonzero_denominator])
        r2[denominator == 0] = 0  # R² is 0 when all y_true values are identical
        
        if multioutput == 'raw_values':
            return r2
        elif multioutput == 'uniform_average':
            return np.mean(r2)
        elif multioutput == 'variance_weighted':
            variance = np.var(y_true, axis=0)
            return np.average(r2, weights=variance) if np.sum(variance) != 0 else np.mean(r2)
        else:
            raise ValueError(f"Unsupported multioutput option: {multioutput}")
    
    @staticmethod
    def adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, 
                         n_features: int, 
                         sample_weight: Optional[np.ndarray] = None) -> float:
        
        r2 = EvaluationMetrics.r2_score(y_true, y_pred, sample_weight)
        n_samples = len(y_true)
        
        # Adjusted R² formula
        return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    
    @staticmethod
    def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray,
                                sample_weight: Optional[np.ndarray] = None,
                                multioutput: str = 'uniform_average') -> Union[float, np.ndarray]:
       
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            weight = sample_weight[:, np.newaxis]
        else:
            weight = 1
        
        y_diff = y_true - y_pred
        y_true_mean = np.average(y_true, axis=0, weights=sample_weight)
        
        # Calculate numerator (variance of the errors)
        numerator = np.average((y_diff - np.average(y_diff, axis=0, weights=sample_weight))**2,
                              axis=0, weights=sample_weight)
        
        # Calculate denominator (variance of y_true)
        denominator = np.average((y_true - y_true_mean)**2, axis=0, weights=sample_weight)
        
        # Avoid division by zero
        nonzero_denominator = denominator != 0
        
        explained_variance = np.zeros(y_true.shape[1])
        explained_variance[nonzero_denominator] = 1 - (numerator[nonzero_denominator] / denominator[nonzero_denominator])
        
        if multioutput == 'raw_values':
            return explained_variance
        elif multioutput == 'uniform_average':
            return np.mean(explained_variance)
        elif multioutput == 'variance_weighted':
            variance = np.var(y_true, axis=0)
            return np.average(explained_variance, weights=variance) if np.sum(variance) != 0 else np.mean(explained_variance)
        else:
            raise ValueError(f"Unsupported multioutput option: {multioutput}")
    
    @staticmethod
    def mean_bias_error(y_true: np.ndarray, y_pred: np.ndarray,
                       sample_weight: Optional[np.ndarray] = None) -> float:
       
        errors = y_pred - y_true
        
        if sample_weight is not None:
            return np.average(errors, weights=sample_weight)
        else:
            return np.mean(errors)
    
    @staticmethod
    def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray,
                             multioutput: str = 'uniform_average') -> Union[float, np.ndarray]:
        
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        errors = np.abs(y_true - y_pred)
        median_errors = np.median(errors, axis=0)
        
        if multioutput == 'raw_values':
            return median_errors
        elif multioutput == 'uniform_average':
            return np.mean(median_errors)
        else:
            raise ValueError(f"Unsupported multioutput option: {multioutput}")
    
    # Visualization methods
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                            normalize: Optional[str] = None,
                            class_names: Optional[List[str]] = None,
                            cmap: str = 'Blues',
                            figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        
        cm = EvaluationMetrics.confusion_matrix(y_true, y_pred, normalize)
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap(cmap))
        ax.figure.colorbar(im, ax=ax)
        
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]
        
        ax.set(xticks=np.arange(cm.shape[1]),
             yticks=np.arange(cm.shape[0]),
             xticklabels=class_names,
             yticklabels=class_names,
             title='Confusion Matrix',
             ylabel='True label',
             xlabel='Predicted label')
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray,
                      class_names: Optional[List[str]] = None,
                      figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if y_score.ndim == 1 or y_score.shape[1] <= 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_score if y_score.ndim == 1 else y_score[:, 1])
            roc_auc = np.trapz(tpr, fpr)
            
            ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            
        else:
            # Multi-class classification
            unique_classes = np.unique(y_true)
            if class_names is None:
                class_names = [str(cls) for cls in unique_classes]
            
            for i, cls in enumerate(unique_classes):
                fpr, tpr, _ = roc_curve((y_true == cls).astype(int), y_score[:, i])
                roc_auc = np.trapz(tpr, fpr)
                
                ax.plot(fpr, tpr, lw=2, 
                       label=f'ROC curve for {class_names[i]} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05],
             title='Receiver Operating Characteristic (ROC)',
             xlabel='False Positive Rate',
             ylabel='True Positive Rate')
        
        ax.legend(loc='lower right')
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray,
                                  class_names: Optional[List[str]] = None,
                                  figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if y_score.ndim == 1 or y_score.shape[1] <= 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_score if y_score.ndim == 1 else y_score[:, 1])
            pr_auc = np.trapz(precision, recall)
            
            ax.plot(recall, precision, lw=2, 
                   label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
            
        else:
            # Multi-class classification
            unique_classes = np.unique(y_true)
            if class_names is None:
                class_names = [str(cls) for cls in unique_classes]
            
            for i, cls in enumerate(unique_classes):
                precision, recall, _ = precision_recall_curve((y_true == cls).astype(int), y_score[:, i])
                pr_auc = np.trapz(precision, recall)
                
                ax.plot(recall, precision, lw=2, 
                       label=f'Precision-Recall curve for {class_names[i]} (AUC = {pr_auc:.2f})')
        
        ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05],
             title='Precision-Recall Curve',
             xlabel='Recall',
             ylabel='Precision')
        
        ax.legend(loc='lower left')
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                      figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Residuals vs. Predicted
        ax1.scatter(y_pred, residuals)
        ax1.axhline(y=0, color='r', linestyle='-')
        ax1.set(title='Residuals vs. Predicted Values',
              xlabel='Predicted',
              ylabel='Residuals')
        
        # Residuals Distribution
        ax2.hist(residuals, bins=30, alpha=0.7, color='b')
        ax2.set(title='Residuals Distribution',
              xlabel='Residuals',
              ylabel='Frequency')
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_regression_results(y_true: np.ndarray, y_pred: np.ndarray,
                               figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the scatter plot of actual vs. predicted
        ax.scatter(y_true, y_pred, alpha=0.7)
        
        # Plot the perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Calculate and display metrics
        mae = EvaluationMetrics.mean_absolute_error(y_true, y_pred)
        mse = EvaluationMetrics.mean_squared_error(y_true, y_pred)
        rmse = EvaluationMetrics.root_mean_squared_error(y_true, y_pred)
        r2 = EvaluationMetrics.r2_score(y_true, y_pred)
        
        metrics_text = f'MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
        
        ax.set(title='Actual vs. Predicted Values',
             xlabel='Actual',
             ylabel='Predicted')
        
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    
    @staticmethod
    def regression_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        
        return {
            'MAE': EvaluationMetrics.mean_absolute_error(y_true, y_pred),
            'MSE': EvaluationMetrics.mean_squared_error(y_true, y_pred),
            'RMSE': EvaluationMetrics.root_mean_squared_error(y_true, y_pred),
            'MAPE': EvaluationMetrics.mean_absolute_percentage_error(y_true, y_pred),
            'R²': EvaluationMetrics.r2_score(y_true, y_pred),
            'Explained Variance': EvaluationMetrics.explained_variance_score(y_true, y_pred),
            'Mean Bias Error': EvaluationMetrics.mean_bias_error(y_true, y_pred),
            'Median Absolute Error': EvaluationMetrics.median_absolute_error(y_true, y_pred)
        }
    
    @staticmethod
    def classification_summary(y_true: np.ndarray, y_pred: np.ndarray, 
                             y_score: Optional[np.ndarray] = None) -> Dict[str, float]:
      
        metrics = {
            'Accuracy': EvaluationMetrics.accuracy(y_true, y_pred),
            'Balanced Accuracy': EvaluationMetrics.balanced_accuracy(y_true, y_pred),
            'Precision': EvaluationMetrics.precision(y_true, y_pred),
            'Recall': EvaluationMetrics.recall(y_true, y_pred),
            'F1 Score': EvaluationMetrics.f1_score(y_true, y_pred),
            'Matthews Correlation Coefficient': EvaluationMetrics.matthews_corrcoef(y_true, y_pred),
            'Cohen\'s Kappa': EvaluationMetrics.cohen_kappa(y_true, y_pred)
        }
        
        if y_score is not None:
            if y_score.ndim == 1 or y_score.shape[1] <= 2:
                metrics['ROC AUC'] = EvaluationMetrics.roc_auc(y_true, y_score if y_score.ndim == 1 else y_score[:, 1])
                metrics['PR AUC'] = EvaluationMetrics.precision_recall_auc(y_true, y_score if y_score.ndim == 1 else y_score[:, 1])
            else:
                metrics['ROC AUC (macro)'] = EvaluationMetrics.roc_auc(y_true, y_score, multi_class='ovr')
        
        return metrics


# Example usage
if __name__ == "__main__":
    # Classification example
    np.random.seed(42)
    y_true_cls = np.random.choice([0, 1, 2], size=100)
    y_pred_cls = np.random.choice([0, 1, 2], size=100)
    y_score_cls = np.random.rand(100, 3)
    
    print("Classification metrics:")
    cls_metrics = EvaluationMetrics.classification_summary(y_true_cls, y_pred_cls, y_score_cls)
    for metric, value in cls_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Regression example
    y_true_reg = np.random.randn(100) * 10
    y_pred_reg = y_true_reg + np.random.randn(100) * 2
    
    print("\nRegression metrics:")
    reg_metrics = EvaluationMetrics.regression_summary(y_true_reg, y_pred_reg)
    for metric, value in reg_metrics.items():
        print(f"{metric}: {value:.4f}")