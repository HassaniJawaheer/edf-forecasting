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
    """Add temporal and external features to 30-min Eco2mix data."""

    def __init__(self, df):
        self.df = df.copy()
        self.df["Datetime"] = pd.to_datetime(self.df["Datetime"])

    def add_hour(self):
        self.df["hour"] = self.df["Datetime"].dt.hour

    def add_dayofweek(self):
        self.df["dayofweek"] = self.df["Datetime"].dt.dayofweek

    def add_dayofyear(self):
        self.df["dayofyear"] = self.df["Datetime"].dt.dayofyear

    def add_month(self):
        self.df["month"] = self.df["Datetime"].dt.month

    def add_year(self):
        self.df["year"] = self.df["Datetime"].dt.year

    def add_weekofyear(self):
        self.df["weekofyear"] = self.df["Datetime"].dt.isocalendar().week.astype(int)

    def add_quarter(self):
        self.df["quarter"] = self.df["Datetime"].dt.quarter

    def add_is_weekend(self):
        self.df["is_weekend"] = self.df["Datetime"].dt.dayofweek.isin([5, 6]).astype(int)

    def add_temperature(self, latitude=48.85, longitude=2.35):
        """Add forecast temperature using Open-Meteo API."""
        self.df["Hour"] = self.df["Datetime"].dt.floor("h")
        start_date = self.df["Hour"].min().strftime("%Y-%m-%d")
        end_date = self.df["Hour"].max().strftime("%Y-%m-%d")

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m",
            "timezone": "Europe/Paris"
        }

        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            temp_df = pd.DataFrame({
                "Hour": pd.to_datetime(data["hourly"]["time"]),
                "temperature_2m": data["hourly"]["temperature_2m"]
            })
            self.df = self.df.merge(temp_df, on="Hour", how="left")
            self.df.drop(columns=["Hour"], inplace=True)
            logging.info("Forecast temperature added.")
        except requests.RequestException as e:
            logging.error(f"Temperature API error: {e}")

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

    def run(self, include=None):
        include = include or []

        if "hour" in include:
            self.add_hour()
        if "dayofweek" in include:
            self.add_dayofweek()
        if "dayofyear" in include:
            self.add_dayofyear()
        if "month" in include:
            self.add_month()
        if "year" in include:
            self.add_year()
        if "weekofyear" in include:
            self.add_weekofyear()
        if "quarter" in include:
            self.add_quarter()
        if "is_weekend" in include:
            self.add_is_weekend()
        if "temperature" in include:
            self.add_temperature()
        if "vacation" in include:
            self.add_vacation()

        return self.df
