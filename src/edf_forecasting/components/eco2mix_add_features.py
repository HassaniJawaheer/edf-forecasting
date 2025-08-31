import pandas as pd
import requests
import logging
import holidays

logging.basicConfig(level=logging.INFO)

class Eco2mixFeaturesDay:
    """Adds selected external features to daily Eco2mix data."""

    def __init__(self, df):
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])

    def add_temperature(self, latitude=48.85, longitude=2.35):
        start_date = self.df['Date'].min().strftime("%Y-%m-%d")
        end_date = self.df['Date'].max().strftime("%Y-%m-%d")
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude, "longitude": longitude,
            "start_date": start_date, "end_date": end_date,
            "daily": "apparent_temperature_min,apparent_temperature_max,apparent_temperature_mean",
            "timezone": "Europe/Paris"
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            temp_df = pd.DataFrame(response.json()["daily"])
            temp_df["Date"] = pd.to_datetime(temp_df["time"])
            temp_df.drop(columns=["time"], inplace=True)
            self.df = self.df.merge(temp_df, on="Date", how="left")
            logging.info("Temperature features added.")
        except requests.RequestException as e:
            logging.error(f"[API Error - Temperature] {e}")

    def add_sunshine(self, latitude=48.85, longitude=2.35):
        start_date = self.df['Date'].min().strftime("%Y-%m-%d")
        end_date = self.df['Date'].max().strftime("%Y-%m-%d")
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude, "longitude": longitude,
            "start_date": start_date, "end_date": end_date,
            "daily": "sunshine_duration",
            "timezone": "Europe/Paris"
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            sun_df = pd.DataFrame(response.json()["daily"])
            sun_df["Date"] = pd.to_datetime(sun_df["time"])
            sun_df.drop(columns=["time"], inplace=True)
            self.df = self.df.merge(sun_df, on="Date", how="left")
            logging.info("Sunshine features added.")
        except requests.RequestException as e:
            logging.error(f"[API Error - Sunshine] {e}")

    def add_weekday(self):
        jours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
        self.df["weekday"] = self.df["Date"].dt.dayofweek.apply(lambda x: jours[x])

    def add_month(self):
        mois = [
            "janvier", "février", "mars", "avril", "mai", "juin",
            "juillet", "août", "septembre", "octobre", "novembre", "décembre"
        ]
        self.df["month"] = self.df["Date"].dt.month.apply(lambda x: mois[x - 1])

    def add_season(self):
        def season(m):
            return (
                "hiver" if m in [12, 1, 2] else
                "printemps" if m in [3, 4, 5] else
                "été" if m in [6, 7, 8] else
                "automne"
            )
        self.df["season"] = self.df["Date"].dt.month.apply(season)

    def add_vacation(self):
        years = self.df["Date"].dt.year.unique()
        public_holidays = holidays.FR(years=years)
        vacation_ranges = {
            "winter": ("02-05", "03-07"), "spring": ("04-09", "05-09"),
            "summer": ("07-06", "08-31"), "Toussaint": ("10-17", "11-02"),
            "christmas": ("12-18", "12-31")
        }
        vacations = set()
        for y in years:
            for start, end in vacation_ranges.values():
                vacations.update(pd.date_range(f"{y}-{start}", f"{y}-{end}").date)
        self.df["is_vacation"] = self.df["Date"].dt.date.apply(
            lambda d: int(d in public_holidays or d in vacations)
        )

    def run(self, include=None):
        include = include or []
        if "temperature" in include:
            self.add_temperature()
        if "sunshine" in include:
            self.add_sunshine()
        if "weekday" in include:
            self.add_weekday()
        if "month" in include:
            self.add_month()
        if "season" in include:
            self.add_season()
        if "vacation" in include:
            self.add_vacation()
        return self.df

class Eco2mixFeaturesMinute:
    """Adds selected external features to 30-min Eco2mix data."""

    def __init__(self, df):
        self.df = df.copy()
        self.df["Datetime"] = pd.to_datetime(self.df["Datetime"])

    def add_temperature(self, latitude=48.85, longitude=2.35):
        self.df["Hour"] = self.df["Datetime"].dt.floor("h")
        start_date = self.df["Hour"].min().strftime("%Y-%m-%d")
        end_date = self.df["Hour"].max().strftime("%Y-%m-%d")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature",
            "timezone": "Europe/Paris"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            temp_df = pd.DataFrame(data["hourly"])
            temp_df["Hour"] = pd.to_datetime(temp_df["time"])
            temp_df.drop(columns=["time"], inplace=True)
            self.df = self.df.merge(temp_df, on="Hour", how="left")
            self.df.drop(columns=["Hour"], inplace=True)

            logging.info("Temperature features added.")
        except requests.RequestException as e:
            logging.error(f"[API Error - Temperature] {e}")

    def add_sunshine(self, latitude=48.85, longitude=2.35):
        self.df["Date"] = self.df["Datetime"].dt.date
        start_date = min(self.df["Date"]).strftime("%Y-%m-%d")
        end_date = max(self.df["Date"]).strftime("%Y-%m-%d")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "sunshine_duration",
            "timezone": "Europe/Paris"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            sun_df = pd.DataFrame(data["daily"])
            sun_df["Date"] = pd.to_datetime(sun_df["time"]).dt.date
            sun_df.drop(columns=["time"], inplace=True)
            self.df = self.df.merge(sun_df, on="Date", how="left")
            self.df.drop(columns=["Date"], inplace=True)
            logging.info("Sunshine features added.")
        except requests.RequestException as e:
            logging.error(f"[API Error - Sunshine] {e}")

    def add_weekday(self):
        self.df["Date"] = self.df["Datetime"].dt.date
        jours_fr = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
        self.df["weekday"] = pd.to_datetime(self.df["Date"]).dt.dayofweek.apply(lambda x: jours_fr[x])
        self.df.drop(columns=["Date"], inplace=True)
        logging.info("Weekday column added.")

    def add_month(self):
        self.df["Date"] = self.df["Datetime"].dt.date
        mois_fr = [
            "janvier", "février", "mars", "avril", "mai", "juin",
            "juillet", "août", "septembre", "octobre", "novembre", "décembre"
        ]
        self.df["month"] = pd.to_datetime(self.df["Date"]).dt.month.apply(lambda x: mois_fr[x - 1])
        self.df.drop(columns=["Date"], inplace=True)
        logging.info("Month column added.")

    def add_season(self):
        self.df["Date"] = self.df["Datetime"].dt.date

        def get_season(month):
            return (
                "hiver" if month in [12, 1, 2] else
                "printemps" if month in [3, 4, 5] else
                "été" if month in [6, 7, 8] else
                "automne"
            )
        self.df["season"] = pd.to_datetime(self.df["Date"]).dt.month.apply(get_season)
        self.df.drop(columns=["Date"], inplace=True)
        logging.info("Season column added.")

    def add_vacation(self):
        self.df["Date"] = self.df["Datetime"].dt.date
        years = pd.to_datetime(self.df["Date"]).dt.year.unique()
        public_holidays = holidays.FR(years=years)

        zone_vacations_by_year = {
            "winter": ("02-05", "03-07"),
            "spring": ("04-09", "05-09"),
            "summer": ("07-06", "08-31"),
            "Toussaint": ("10-17", "11-02"),
            "christmas": ("12-18", "12-31")
        }

        vacations = set()
        for year in years:
            for start, end in zone_vacations_by_year.values():
                vacations.update(pd.date_range(start=f"{year}-{start}", end=f"{year}-{end}").date)

        self.df["is_vacation"] = self.df["Date"].apply(lambda d: int(d in public_holidays or d in vacations))
        self.df.drop(columns=["Date"], inplace=True)
        logging.info("Vacation flag added.")

    def run(self, include=None):
        include = include or []
        if "temperature" in include:
            self.add_temperature()
        if "sunshine" in include:
            self.add_sunshine()
        if "weekday" in include:
            self.add_weekday()
        if "month" in include:
            self.add_month()
        if "season" in include:
            self.add_season()
        if "vacation" in include:
            self.add_vacation()
        return self.df
