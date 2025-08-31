import pandas as pd

class Eco2mixCleaner:
    def __init__(self,
        columns_to_keep,
        tempo_column_name='Type de jour Tempo',
        new_tempo_column_name='tempo',
        consumption_col='Consommation'
    ):
        self.columns_to_keep = columns_to_keep
        self.tempo_column_name = tempo_column_name
        self.new_tempo_column_name = new_tempo_column_name
        self.consumption_col = consumption_col

    def clean_definitive(self, df):
        df = df.dropna(subset=[self.consumption_col])
        return df[self.columns_to_keep]

    def clean_tempo(self, df):
        df = self._fill_missing_dates(df)
        df[self.tempo_column_name] = self._fill_missing_values(df, self.tempo_column_name)
        return df.rename(columns={self.tempo_column_name: self.new_tempo_column_name})

    def _fill_missing_dates(self, df):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        for idx in range(1, len(df)):
            if pd.isna(df.at[idx, 'Date']) and not pd.isna(df.at[idx - 1, 'Date']):
                df.at[idx, 'Date'] = df.at[idx - 1, 'Date'] + pd.Timedelta(days=1)
        return df

    def _fill_missing_values(self, df, column):
        return df[column].ffill()
