import pandas as pd

df = pd.read_csv("data/raw_taxi_data.csv")

cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "fare_amount", "passenger_count"]
df = df[cols].dropna()

df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 50)]
df = df[(df["fare_amount"] > 2) & (df["fare_amount"] < 200)]

df.to_csv("data/cleaned_taxi_data.csv", index=False)
print(f"Cleaned data saved with {len(df)} records.")
