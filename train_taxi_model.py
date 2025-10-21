import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib, json

df = pd.read_csv("data/cleaned_taxi_data.csv")

pickup = pd.to_datetime(df["tpep_pickup_datetime"])
dropoff = pd.to_datetime(df["tpep_dropoff_datetime"])
df["trip_duration_min"] = (dropoff - pickup).dt.total_seconds() / 60

X = df[["fare_amount", "passenger_count", "trip_duration_min"]]
y = df["trip_distance"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, "taxi_model.pkl")

mse = mean_squared_error(y_test, model.predict(X_test))
with open("taxi_metrics.json", "w") as f:
    json.dump({"mse": mse}, f, indent=4)

print(f"Model trained successfully — MSE = {mse:.4f}")
