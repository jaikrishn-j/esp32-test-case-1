import json
import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta


MQTT_BROKER = "broker.emqx.io"  # Change to your EMQX host
MQTT_PORT = 1883
MQTT_TOPIC = "agriculture/crop_data"

# Storage for the last results
last_rest_response = {"message": "No REST data processed yet"}
last_mqtt_response = {"message": "No MQTT data processed yet"}

# Load resources once
MODEL = joblib.load("model/crop_prediction_model.pkl")
CROP_DF = pd.read_csv("datasets/crop_recommendation_with_yield_simple.csv")
PRICE_DF = pd.read_csv("datasets/CropPrice.csv")

app = FastAPI(title="Smart Agriculture API")



def get_coordinates(location):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
    res = requests.get(url).json()
    return (res["results"][0]["latitude"], res["results"][0]["longitude"]) if "results" in res else (None, None)

def get_rainfall(lat, lon, days):
    end = datetime.today()
    start = end - timedelta(days=days)
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start.strftime('%Y-%m-%d')}&end_date={end.strftime('%Y-%m-%d')}&daily=precipitation_sum&timezone=auto"
    data = requests.get(url).json()
    return round(sum(data["daily"]["precipitation_sum"]), 2) if "daily" in data else 0

def process_recommendation(data_dict):
    lat, lon = get_coordinates(data_dict['location'])
    if lat is None: return {"error": "Location not found"}
    
    seasonal_rain = get_rainfall(lat, lon, 120)
    
    # ML Prediction
    input_df = pd.DataFrame([[data_dict['temp'], data_dict['hum'], seasonal_rain, data_dict['ph']]], 
                            columns=['temperature', 'humidity', 'rainfall', 'ph'])
    ml_crop = MODEL.predict(input_df)[0]
    
    # Profit Calc
    ideal = CROP_DF.groupby("label")[['temperature', 'humidity', 'rainfall', 'ph']].mean().reset_index()
    merged = pd.merge(ideal, PRICE_DF, on="label")
    
    def calc_suitability(row):
        s = [max(0, 1 - abs(data_dict['temp'] - row["temperature"]) / 20),
             max(0, 1 - abs(data_dict['hum'] - row["humidity"]) / 50),
             max(0, 1 - abs(seasonal_rain - row["rainfall"]) / 200),
             max(0, 1 - abs(data_dict['ph'] - row["ph"]) / 3)]
        return sum(s) / 4

    merged["Suitability"] = merged.apply(calc_suitability, axis=1)
    best = merged.loc[(merged["Suitability"] * merged["base_yield"] * merged["price_per_ton_inr"]).idxmax()]

    return {
        "timestamp": datetime.now().isoformat(),
        "ml_crop": ml_crop,
        "profit_crop": best['label'],
        "suitability": round(best['Suitability'], 3),
        "rainfall": seasonal_rain
    }



def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker with result code {rc}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    global last_mqtt_response
    try:
        payload = json.loads(msg.payload.decode())
        # Expecting JSON: {"location": "...", "temp": 27, "hum": 80, "ph": 6, "moisture": 2500}
        result = process_recommendation(payload)
        last_mqtt_response = result
        print("MQTT Data Updated")
    except Exception as e:
        print(f"MQTT Processing Error: {e}")

mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()


class PredictionRequest(BaseModel):
    location: str
    temp: float
    hum: float
    ph: float
    moisture: int

@app.post("/predict")
async def predict_via_rest(req: PredictionRequest):
    global last_rest_response
    result = process_recommendation(req.dict())
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    last_rest_response = result
    return result

@app.get("/data/rest")
async def get_last_rest():
    return {"source": "REST API", "data": last_rest_response}

@app.get("/data/mqtt")
async def get_last_mqtt():
    return {"source": "MQTT EMQX", "data": last_mqtt_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)