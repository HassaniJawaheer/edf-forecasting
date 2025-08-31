import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class Eco2MixAddTempo:
    """Class for adding tempo information to Eco2mix data."""

    def __init__(self, mode='aggregate_day'):
        self.mode = mode

    def add_tempo(self, df_data, df_tempo):
        if self.mode == 'aggregate_day':
            return self._add_tempo_day(df_data, df_tempo)
        elif self.mode == 'aggregate_minute':
            return self._add_tempo_minute(df_data, df_tempo)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _add_tempo_day(self, df_data, df_tempo):
        df_tempo['Date'] = pd.to_datetime(df_tempo['Date']).dt.date
        df_data['Date'] = pd.to_datetime(df_data['Date']).dt.date
        return df_data.merge(df_tempo[['Date', 'tempo']], how='left', on='Date')

    def _add_tempo_minute(self, df_data, df_tempo):
        df_tempo['Date'] = pd.to_datetime(df_tempo['Date']).dt.date
        df_data['date'] = pd.to_datetime(df_data['Datetime']).dt.date
        df_final = df_data.merge(df_tempo[['Date', 'tempo']], how='left', left_on='date', right_on='Date')
        return df_final.drop(columns=['date', 'Date'])
