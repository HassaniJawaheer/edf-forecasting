import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class Eco2mixAggregate:
    """Class for aggregating Eco2mix 30-min data into daily vectors."""

    def __init__(self):
        pass

    def aggregate(self, df):
        """Aggregate 30-min data into daily vectors."""
        if 'Datetime' not in df.columns:
            logging.error("Missing 'Datetime' column in definitive data.")
            raise ValueError("Missing 'Datetime' column in definitive data.")

        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df['Date'] = df['Datetime'].dt.date
        df['HourMinute'] = df['Datetime'].dt.strftime('%Hh%M')

        df = df.drop(columns=['Datetime'])

        feature_cols = [col for col in df.columns if col not in ['Date', 'HourMinute']]

        daily_records = []

        for date, group in df.groupby('Date'):
            flat_record = {'Date': date}
            group = group.reset_index(drop=True)

            for idx, row in group.iterrows():
                time_label = row['HourMinute']
                for feature in feature_cols:
                    flat_record[f"{feature}_{time_label}"] = row[feature]

            daily_records.append(flat_record)

        df_final = pd.DataFrame(daily_records)
        return df_final
