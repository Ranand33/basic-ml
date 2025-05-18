import numpy as np
import pandas as pd
from typing import Union, Optional, Literal

class DataNormalizer:

    def __init__(self):
        # Initialize the DataNormalizer class
        self.fitted_params = {}

    def fit(self, data: Union[np.ndarray, pd.DataFrame, pd.Series, list],
            method: Literal['minmax', 'z-score', 'decimal', 'robust'] = 'min-max',
            feature_range: tuple = (0,1)):
        
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)

        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1,1)

        self.fitted_params['method'] = method
        self.fitted_params['feature_range'] = feature_range

        if method == 'minmax':
            self.fitted_params['min'] = np.min(data, axis=0)
            self.fitted_params['max'] = np.max(data, axis=0)
        
        elif method == 'z-score':
            self.fitted_params['mean'] = np.mean(data, axis=0)
            self.fitted_params['std'] = np.std(data, axis=0)
        
        elif method == 'decimal':
            self.fitted_params['max_abs'] = np.max(np.abs(data), axis=0)
            self.fitted_params['scaling_factor'] = 10 * np.ceil(np.log10(self.fitted_params['max_abs'] + 1e-10))
        
        elif method == 'robust':
            self.fitted_params['median'] = np.median(data, axis=0)
            q1 = np.percentile(data, 25, axis=0)
            q3 = np.percentile(data, 75, axis=0)
            self.fitted_params['iqr'] = q3 - q1
        else:
            raise ValueError(f"Invalid normalization method: {method}")
        
        return self
    
    def transform(self, data: Union[np.ndarray, pd.DataFrame, pd.Series, list]) -> np.ndarray:

        if not self.fitted_params:
            raise ValueError("Normalizer has not been fitted yet. Call fit() first.")
        
        if isinstance(data, (pd.DataFrame, pd.Series)):
            is_df = isinstance(data, pd.DataFrame)
            index = data.index 
            columns = data.columns if is_df else None
            data = data.values
        
        elif isinstance(data, list):
            data = np.array(data)
            is_df = False
            index = None
            columns = None
        else:
            is_df = False
            index = None
            columns = None
        
        if data.ndim == 1:
            data = data.reshape(-1,1)
        
        method = self.fitted_params['method']
        feature_range = self.fitted_params['feature_range']

        if method == 'minmax':
            min_val = self.fitted_params['min']
            max_val = self.fitted_params['max']
            feature_range = self.fitted_params['feature_range']

            range_data = max_val - min_val
            range_data = np.where(range_data == 0, 1, range_data)

            normalized_data = (data - min_val) / range_data
            normalized_data = normalized_data * (feature_range[1] - feature_range[0]) + feature_range[0]
        
        elif method == 'z-score':
            mean = self.fitted_params['mean']
            std = self.fitted_params['std']

            std = np.where(std == 0, 1, std)

            normalized_data = (data - mean) / std
        
        elif method == 'decimal':
            scaling_factor = self.fitted_params['scaling_factor']
            normalized_data = data / scaling_factor

        elif method == 'robust':
            median = self.fitted_params['median']
            iqr = self.fitted_params['iqr']

            iqr = np.where(iqr == 0, 1, iqr)

            normalized_data = (data - median) / iqr

        if is_df:
            normalized_data = pd.DataFrame(normalized_data, index=index, columns=columns)
        elif index is not None:
            normalized_data = pd.Series(normalized_data.ravel(), index=index)
        
        return normalized_data
    
    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame, pd.Series, list],
                  method: str = 'min-max',
                  feature_range: tuple = (0,1)) -> np.ndarray:
        return self.fit(data, method, feature_range).transform(data)

    def inverse_transform(self, data: Union[np.ndarray, pd.DataFrame, pd.Series, list]) -> np.ndarray:
        if not self.fitted_params:
            raise ValueError("Normalizer has not been fitted yet. Call fit() first.")
    
        if isinstance(data, (pd.DataFrame, pd.Series)):
            is_df = isinstance(data, pd.DataFrame)
            index = data.index
            columns = data.columns if is_df else None
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)
            is_df = False
            index = None
            columns = None
        else:
            is_df = False
            index = None
            columns = None
    
        if data.ndim == 1:
            data = data.reshape(-1,1)
    
        method = self.fitted_params['method']
    
        if method == 'minmax':
            min_val = self.fitted_params['min']
            max_val = self.fitted_params['max']
            feature_range = self.fitted_params['feature_range']

            original_data = (data - feature_range[0]) / (feature_range[1] - feature_range[0])
            original_data = original_data * (max_val - min_val) + min_val
    
        elif method == 'z-score':
            mean = self.fitted_params['mean']
            std = self.fitted_params['std']

            original_data = data * std + mean

        elif method == 'decimal':
            scaling_factor = self.fitted_params['scaling_factor']
            original_data = data * scaling_factor
    
        elif method == 'robust':
            median = self.fitted_params['median']
            iqr = self.fitted_params['iqr']
            original_data = data * iqr + median
    
        if is_df:
            original_data = pd.DataFrame(original_data, index=index, columns=columns)
        elif index is not None:
            original_data = pd.Series(original_data.ravel(), index=index)
    
        return original_data

if __name__ == "__main__":

    import numpy as np
    np.random.seed(42)
    data = np.random.randn(100, 3) * 10 + np.array([5, -5, 0])

    normalizer = DataNormalizer()

    normalized_min_max = normalizer.fit_transform(data, method='minmax')
    print("Min-Max normalized data (first 5 rows):")
    print(normalized_min_max[:5])

    normalizer = DataNormalizer()
    normalized_z_score = normalizer.fit_transform(data, method='z-score')
    print("\nZ-Score normalized data (first 5 rows):")
    print(normalized_z_score[:5])

    normalizer = DataNormalizer()
    normalized_robust = normalizer.fit_transform(data, method='robust')
    print("\nRobust normalized data (first 5 rows):")
    print(normalized_robust[:5])

    import pandas as pd
    df = pd.DataFrame(data, columns=['Feature 1', 'Feature 2', 'Feature 3'])

    normalizer = DataNormalizer()
    normalized_df = normalizer.fit_transform(df, method='minmax')
    print("\nMin-Max normalized DataFrame (first 5 rows):")
    print(normalized_df.head())