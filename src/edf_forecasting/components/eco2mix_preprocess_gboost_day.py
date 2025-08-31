import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

class Eco2mixPreprocessGBoostDay:
    """Prepares daily Eco2mix data for XGBoost training using sliding windows."""

    def __init__(self, 
        target_col_prefix='Consommation_',
        window_size=7,
        seed=42,
        features_to_include=None,
        target_features_to_include=None,
        test_size=0.2,
        shuffle=False
    ):
        self.target_col_prefix = target_col_prefix
        self.window_size = window_size
        self.seed = seed
        self.features_to_include = features_to_include or []
        self.target_features_to_include = target_features_to_include or []
        self.test_size = test_size
        self.shuffle = shuffle

    def run(self, df):
        df, _ = self._encode_categorical(df, self.features_to_include + self.target_features_to_include)

        X, y, _ = self._create_dataset_sliding(
            df,
            self.window_size,
            self.target_col_prefix,
            'Date',
            self.features_to_include,
            self.target_features_to_include
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=self.shuffle, random_state=self.seed
        )

        return pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(y_train), pd.DataFrame(y_test)

    def _create_dataset_sliding(self, df, window_size, target_col_prefix, date_col,
                                features_to_include, target_features_to_include):
        target_cols = [col for col in df.columns if col.startswith(target_col_prefix)]
        X_list, y_list, date_list = [], [], []

        for i in range(window_size, len(df)):
            window_df = df.iloc[i - window_size:i]
            features = []

            for j in range(window_size):
                row = window_df.iloc[j]
                if features_to_include:
                    features.extend(row[features_to_include].tolist())
                features.extend(row[target_cols].tolist())

            if target_features_to_include:
                features.extend(df.iloc[i][target_features_to_include].tolist())

            y_values = df.iloc[i][target_cols].tolist()
            X_list.append(np.array(features))
            y_list.append(y_values)
            date_list.append(df.iloc[i][date_col])

        return np.array(X_list), np.array(y_list), date_list

    def _encode_categorical(self, df, columns, start_at=0):
        df = df.copy()
        mappings = {}

        for col in columns:
            if self._is_boolean_column(df, col):
                df[col] = df[col].astype(int)
            else:
                unique_vals = sorted(df[col].dropna().unique())
                map_dict = {val: i + start_at for i, val in enumerate(unique_vals)}
                df[col] = df[col].map(map_dict)
                mappings[col] = map_dict
        return df, mappings

    def _is_boolean_column(self, df, col):
        unique_vals = set(df[col].dropna().unique())
        return unique_vals <= {0, 1} or unique_vals <= {True, False}
