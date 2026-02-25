import pandas as pd
import numpy as np

def get_crop_recommendation(
    location_name, 
    sensor_temp, 
    sensor_humidity, 
    sensor_ph, 
    sensor_moisture_adc, 
    model, 
    crop_df, 
    price_df
):
    """
    All inputs are passed via arguments. 
    Returns a dictionary containing the prediction and profit analysis.
    """
    # 1. Fetch Weather Data (Using your existing helper functions)
    lat, lon = get_coordinates(location_name)
    if lat is None:
        return {"error": "Location not found"}

    recent_30d = get_rainfall_stats(lat, lon, days=30)
    seasonal_120d = get_rainfall_stats(lat, lon, days=120)
    rain_class = classify_density(recent_30d)

    # 2. Soil Moisture Status
    if sensor_moisture_adc < 2000:
        moisture_status = "Soil too wet, Delay Planting"
    elif sensor_moisture_adc > 3200:
        moisture_status = "Soil too dry, Requires Watering"
    else:
        moisture_status = "Soil Moisture Ideal For Planting"

    # 3. Machine Learning Prediction
    input_df = pd.DataFrame(
        [[sensor_temp, sensor_humidity, seasonal_120d, sensor_ph]], 
        columns=['temperature', 'humidity', 'rainfall', 'ph']
    )
    ml_predicted_crop = model.predict(input_df)[0]

    # 4. Profit-Based Calculation
    ideal_climate = crop_df.groupby("label")[['temperature', 'humidity', 'rainfall', 'ph']].mean().reset_index()
    data = pd.merge(ideal_climate, price_df, on="label")

    def calculate_suitability(row):
        scores = [
            max(0, 1 - abs(sensor_temp - row["temperature"]) / 20),
            max(0, 1 - abs(sensor_humidity - row["humidity"]) / 50),
            max(0, 1 - abs(seasonal_120d - row["rainfall"]) / 200),
            max(0, 1 - abs(sensor_ph - row["ph"]) / 3)
        ]
        return sum(scores) / 4

    data["Suitability"] = data.apply(calculate_suitability, axis=1)
    data["Expected_Yield"] = data["base_yield"] * data["Suitability"]
    data["Estimated_Profit"] = data["Expected_Yield"] * data["price_per_ton_inr"]

    best_crop_row = data.loc[data["Estimated_Profit"].idxmax()]

    # 5. Return Results Object
    return {
        "location": location_name,
        "weather": {
            "recent_rainfall_30d": recent_30d,
            "seasonal_rainfall_120d": seasonal_120d,
            "condition": rain_class
        },
        "soil_status": moisture_status,
        "ml_recommendation": ml_predicted_crop,
        "profit_recommendation": {
            "crop": best_crop_row['label'],
            "suitability": round(best_crop_row['Suitability'], 3),
            "estimated_profit": round(best_crop_row['Estimated_Profit'], 2)
        }
    }