import numpy as np

class XGBCalibrator30min:
    def __init__(self, df_cal, model, error_type, windows_size, target_col):
        self.model = model
        self.df_cal = df_cal
        self.error_type = error_type
        self.windows_size = windows_size
        self.target_col = target_col
        self.q_inf = None
        self.q_sup = None
        self.X_cal = None
        self.y_cal = None

    def _create_windows(self):
        if self.target_col not in self.df_cal.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in calibration dataframe.")

        values = self.df_cal[self.target_col].values

        if len(values) <= self.windows_size:
            raise ValueError("Insufficient data to create at least one window.")

        self.X_cal = np.lib.stride_tricks.sliding_window_view(values, window_shape=self.windows_size)[:-1]
        self.y_cal = values[self.windows_size:]

    def run(self, alpha=0.05):
        self._create_windows()

        y_pred = self.model.predict(self.X_cal)
        errors = self.y_cal - y_pred

        if self.error_type == "absolute":
            errors = np.abs(errors)

        self.q_inf = np.quantile(errors, alpha / 2)
        self.q_sup = np.quantile(errors, 1 - alpha / 2)

        return float(self.q_inf), float(self.q_sup)

