import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

# Load trained model and feature names
model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")


# Input schema
class HouseFeatures(BaseModel):

    MedInc: float = Field(
        gt=0,
        description="Median income of neighbourhood"
    )

    HouseAge: float = Field(
        gt=0,
        description="Average age of houses in the block"
    )

    AveRooms: float = Field(
        gt=0,
        description="Average number of rooms"
    )

    AveBedrms: float = Field(
        gt=0,
        description="Average number of bedrooms"
    )

    Population: float = Field(
        gt=0,
        description="Total population"
    )

    AveOccup: float = Field(
        gt=0,
        description="Average household occupancy"
    )

    Latitude: float = Field(
        gt=32,
        le=42,
        description="Latitude"
    )

    Longitude: float = Field(
        gt=-125,
        le=-114,
        description="Longitude"
    )


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "California House Prediction API",
        "status": "running",
        "endpoint": "Send POST request to /predict"
    }


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "RandomForestRegressor",
        "features": features,
        "avg_error": "$39,000"
    }


# Prediction endpoint
@app.post("/predict")
def predict(house: HouseFeatures):

    try:

        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])

        # Maintain training feature order
        input_data = input_data[features]

        # Predict house price
        predicted = model.predict(input_data)[0]

        # Convert into dollars
        price_usd = predicted * 100000

        return {
            "predicted_price": f"${price_usd:,.0f}",

            "predicted_price_short":
                f"${predicted:.2f} hundred thousands",

            "finance_range":
                f"${price_usd - 39000:,.0f} to ${price_usd + 39000:,.0f}"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )