import requests
import pandas as pd
from pathlib import Path
import os

# =========================
# Basic settings
# =========================

# Current Python file folder
CURRENT_FOLDER = Path(__file__).parent

# Output folder for generated CSV files
OUTPUT_FOLDER = CURRENT_FOLDER / "CensusData"
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Census API key
# IMPORTANT: Do not upload your real API key to GitHub.
# Option 1: paste your key locally
API_KEY = "YOUR_API_KEY"

# Milwaukee County, Wisconsin
STATE = "55"
COUNTY = "079"
YEAR = "2024"


# =========================
# Functions
# =========================

def fetch_acs_profile_data(year, variables, state, county, api_key):
    """
    Download ACS 5-year profile data at census tract level.
    """
    url = f"https://api.census.gov/data/{year}/acs/acs5/profile"

    params = {
        "get": ",".join(["NAME"] + variables),
        "for": "tract:*",
        "in": f"state:{state} county:{county}",
        "key": api_key
    }

    response = requests.get(url, params=params)

    print("Status code:", response.status_code)

    if response.status_code != 200:
        print("Request failed:")
        print(response.text)
        return pd.DataFrame()

    data = response.json()

    df = pd.DataFrame(data[1:], columns=data[0])

    return df


def clean_negative_missing_values(df, columns):
    """
    Census sometimes uses negative values like -666666666 for missing data.
    This function converts those values to missing values.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = pd.NA

    return df


def save_dataframe(df, file_name):
    """
    Save dataframe into the CensusData folder.
    """
    file_path = OUTPUT_FOLDER / file_name
    df.to_csv(file_path, index=False)

    print("Data saved to:")
    print(file_path)


def download_acs_income_data():
    """
    Download ACS median household income data for Milwaukee County census tracts.
    """
    variables = [
        "DP03_0062E"   # Median household income
    ]

    df = fetch_acs_profile_data(
        year=YEAR,
        variables=variables,
        state=STATE,
        county=COUNTY,
        api_key=API_KEY
    )

    if df.empty:
        print("No income data downloaded.")
        return

    df = df.rename(columns={
        "DP03_0062E": "median_household_income"
    })

    df = clean_negative_missing_values(
        df,
        columns=["median_household_income"]
    )

    save_dataframe(df, "acs_income_mke_tract_2024.csv")


def download_acs_demographic_data():
    """
    Download ACS demographic data for Milwaukee County census tracts.
    """
    variables = [
        "DP05_0001E",    # Total population
        "DP05_0077E",    # Hispanic or Latino population
        "DP05_0079E",    # Non-Hispanic White population
        "DP05_0079PE"    # Percent non-Hispanic White
    ]

    df = fetch_acs_profile_data(
        year=YEAR,
        variables=variables,
        state=STATE,
        county=COUNTY,
        api_key=API_KEY
    )

    if df.empty:
        print("No demographic data downloaded.")
        return

    df = df.rename(columns={
        "DP05_0001E": "total_population",
        "DP05_0077E": "hispanic_or_latino_population",
        "DP05_0079E": "non_hispanic_white_population",
        "DP05_0079PE": "percent_non_hispanic_white"
    })

    df = clean_negative_missing_values(
        df,
        columns=[
            "total_population",
            "hispanic_or_latino_population",
            "non_hispanic_white_population",
            "percent_non_hispanic_white"
        ]
    )

    save_dataframe(df, "acs_demo_mke_tract_2024.csv")


def main():
    download_acs_income_data()
    download_acs_demographic_data()


if __name__ == "__main__":
    main()