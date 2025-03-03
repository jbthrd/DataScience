from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from typing import List, Union
import uvicorn
import datetime as dt

# Initialize FastAPI app
app = FastAPI(
    title="Car Price Prediction API",
    description="API for predicting car prices using XGBoost model",
    version="1.0.0"
)

# Load the saved model
try:
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define input data model based on your training data
class CarFeatures(BaseModel):
    Manufacturer: str
    Model: str
    Category: str
    Leather_interior: str  # "yes" or "no"
    Mileage: str  # Will be processed to extract numeric value
    Cylinders: int
    Engine_volume: str  # Will be processed to extract numeric value and check for Turbo
    Doors: str  # Could be "4", "02-Mar", "04-May", or ">5"
    Wheel: str  # e.g., "left wheel", "right-hand drive"
    Color: str
    Airbags: int
    Prod_year: int
    Drive_wheels: str  # "front", "rear", or "4wd"
    Gear_box_type: str  # "automatic", "manual", "tiptronic", "variator"
    Fuel_type: str  # "gasoline", "diesel", "cng", "hybrid", "lpg", "plugin hybrid"
    
    class Config:
        schema_extra = {
            "example": {
                "Manufacturer": "Toyota",
                "Model": "Camry",
                "Category": "Sedan",
                "Leather_interior": "yes",
                "Mileage": "35000 km",
                "Cylinders": 4,
                "Engine_volume": "2.5 Turbo",
                "Doors": "4",
                "Wheel": "Left wheel",
                "Color": "Black",
                "Airbags": 6,
                "Prod_year": 2018,
                "Drive_wheels": "front",
                "Gear_box_type": "automatic",
                "Fuel_type": "gasoline"
            }
        }

# Define output data model
class PredictionResponse(BaseModel):
    predicted_price: float

# Process input data to match exactly with the training features
def process_input(input_data: CarFeatures):
    # Convert to dictionary
    data = input_data.dict()
    
    # 1. Convert all string values to lowercase (as done in training)
    for key, value in data.items():
        if isinstance(value, str):
            data[key] = value.lower()
    
    # 2. Process Mileage: Extract numeric value
    mileage = int(data["Mileage"].split()[0])
    
    # 3. Process Engine volume and check for Turbo
    engine_str = data["Engine_volume"]
    turbo = "turbo" in engine_str.lower()
    engine_volume = float(engine_str.split()[0])
    
    # 4. Process Doors
    doors_str = data["Doors"]
    if doors_str == "04-May":
        doors = 4
    elif doors_str == "02-Mar":
        doors = 2
    elif doors_str == ">5":
        doors = 5
    else:
        doors = int(doors_str)
    
    # 5. Convert production year to car age (as done in training)
    current_year = dt.datetime.now().year
    car_age = current_year - data["Prod_year"]
    
    # 6. Create feature array exactly matching training features
    # The order matters and must match: ['Manufacturer', 'Model', 'Prod. year', 'Category', 'Leather interior', 
    # 'Engine volume', 'Mileage', 'Cylinders', 'Doors', 'Wheel', 'Color', 'Airbags', 'Turbo', 
    # 'Drive_4x4', 'Drive_front', 'Drive_rear', 'Gear_automatic', 'Gear_manual', 'Gear_tiptronic', 
    # 'Gear_variator', 'Fuel_cng', 'Fuel_diesel', 'Fuel_hybrid', 'Fuel_hydrogen', 'Fuel_lpg', 
    # 'Fuel_petrol', 'Fuel_plug-in hybrid']
    
    features = [
        hash(data["Manufacturer"]) % 100,          # Manufacturer encoding
        hash(data["Model"]) % 100,                 # Model encoding
        car_age,                                   # Prod. year (as car age)
        hash(data["Category"]) % 100,              # Category encoding
        1 if data["Leather_interior"].lower() == "yes" else 0,  # Leather interior
        engine_volume,                             # Engine volume
        mileage,                                   # Mileage
        data["Cylinders"],                         # Cylinders
        doors,                                     # Doors
        hash(data["Wheel"]) % 100,                 # Wheel encoding
        hash(data["Color"]) % 100,                 # Color encoding
        data["Airbags"],                           # Airbags
        1 if turbo else 0,                         # Turbo
        
        # Drive wheels (one-hot encoded)
        1 if data["Drive_wheels"].lower() == "4wd" else 0,       # Drive_4x4
        1 if data["Drive_wheels"].lower() == "front" else 0,     # Drive_front
        1 if data["Drive_wheels"].lower() == "rear" else 0,      # Drive_rear
        
        # Gear box type (one-hot encoded)
        1 if data["Gear_box_type"].lower() == "automatic" else 0,  # Gear_automatic
        1 if data["Gear_box_type"].lower() == "manual" else 0,     # Gear_manual
        1 if data["Gear_box_type"].lower() == "tiptronic" else 0,  # Gear_tiptronic
        1 if data["Gear_box_type"].lower() == "variator" else 0,   # Gear_variator
        
        # Fuel type (one-hot encoded)
        1 if data["Fuel_type"].lower() == "cng" else 0,          # Fuel_cng
        1 if data["Fuel_type"].lower() == "diesel" else 0,       # Fuel_diesel
        1 if data["Fuel_type"].lower() == "hybrid" else 0,       # Fuel_hybrid
        0,                                                       # Fuel_hydrogen (not in input, set to 0)
        1 if data["Fuel_type"].lower() == "lpg" else 0,          # Fuel_lpg
        1 if data["Fuel_type"].lower() == "gasoline" else 0,     # Fuel_petrol (gasoline)
        1 if data["Fuel_type"].lower() == "plugin hybrid" else 0  # Fuel_plug-in hybrid
    ]
    
    return np.array([features])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Car Price Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_price(car_features: CarFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Process input data
        processed_features = process_input(car_features)
        
        # Make prediction
        prediction = model.predict(processed_features)
        
        # Return prediction
        return {"predicted_price": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)