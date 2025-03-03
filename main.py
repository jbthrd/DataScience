#We import relevant librairies for the exercice
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn
import datetime as dt

#FastAPI creation into app variable
app = FastAPI(
    title="Car Price Prediction API",
    description="API for predicting car prices using XGBoost model",
    version="1.0.0"
)

#Load the best-performing model saved at the end of car_prices
try:
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

#Define input data model
class CarFeatures(BaseModel):
    Manufacturer: str
    Model: str
    Category: str
    Leather_interior: str
    Mileage: str
    Cylinders: int
    Engine_volume: str
    Doors: str
    Wheel: str
    Color: str
    Airbags: int
    Prod_year: int
    Drive_wheels: str
    Gear_box_type: str
    Fuel_type: str
    
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

#Output data (a float) for API responses
class PredictionResponse(BaseModel):
    predicted_price: float

#Process input data and format it for our model
def process_input(input_data: CarFeatures):
    #Convert input into dictionary
    data = input_data.dict()
    
    #Convert string values to lowercase for consistency
    for key, value in data.items():
        if isinstance(value, str):
            data[key] = value.lower()
    
    #Extract numeric value from Mileage
    mileage = int(data["Mileage"].split()[0])
    
    #Extract numeric engine volume and check for turbo
    engine_str = data["Engine_volume"]
    turbo = "turbo" in engine_str.lower()
    engine_volume = float(engine_str.split()[0])
    
    #From doors field to an integer format
    doors_str = data["Doors"]
    if doors_str == "04-May":
        doors = 4
    elif doors_str == "02-Mar":
        doors = 2
    elif doors_str == ">5":
        doors = 5
    else:
        doors = int(doors_str)
    
    #Calculate car age from production year
    current_year = dt.datetime.now().year
    car_age = current_year - data["Prod_year"]
    
    # Create feature array that is exactly matching training format
    # The order matters and must match: ['Manufacturer', 'Model', 'Prod. year', 'Category', 'Leather interior', 
    # 'Engine volume', 'Mileage', 'Cylinders', 'Doors', 'Wheel', 'Color', 'Airbags', 'Turbo', 
    # 'Drive_4x4', 'Drive_front', 'Drive_rear', 'Gear_automatic', 'Gear_manual', 'Gear_tiptronic', 
    # 'Gear_variator', 'Fuel_cng', 'Fuel_diesel', 'Fuel_hybrid', 'Fuel_hydrogen', 'Fuel_lpg', 
    # 'Fuel_petrol', 'Fuel_plug-in hybrid']
    
    features = [
        hash(data["Manufacturer"]) % 100,
        hash(data["Model"]) % 100,
        car_age,
        hash(data["Category"]) % 100,
        1 if data["Leather_interior"].lower() == "yes" else 0,
        engine_volume,
        mileage,
        data["Cylinders"],
        doors,
        hash(data["Wheel"]) % 100,
        hash(data["Color"]) % 100,
        data["Airbags"],
        1 if turbo else 0,
        
        #One-hot encoded drive wheels
        1 if data["Drive_wheels"].lower() == "4wd" else 0,
        1 if data["Drive_wheels"].lower() == "front" else 0,
        1 if data["Drive_wheels"].lower() == "rear" else 0,
        
        #One-hot encoded gear box type
        1 if data["Gear_box_type"].lower() == "automatic" else 0,
        1 if data["Gear_box_type"].lower() == "manual" else 0,
        1 if data["Gear_box_type"].lower() == "tiptronic" else 0,
        1 if data["Gear_box_type"].lower() == "variator" else 0,
        
        #One-hot encoded fuel type
        1 if data["Fuel_type"].lower() == "cng" else 0,
        1 if data["Fuel_type"].lower() == "diesel" else 0,
        1 if data["Fuel_type"].lower() == "hybrid" else 0,
        0,                                                       #Fuel_hydrogen (not in input, set to 0)
        1 if data["Fuel_type"].lower() == "lpg" else 0,
        1 if data["Fuel_type"].lower() == "gasoline" else 0,
        1 if data["Fuel_type"].lower() == "plugin hybrid" else 0
    ]
    
    return np.array([features])

#API root endpoint and welcome message
@app.get("/")
def read_root():
    return {"message": "Welcome to the Car Price Prediction API"}

#Endpoint for predicting car prices
@app.post("/predict", response_model=PredictionResponse)
def predict_price(car_features: CarFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        #Process input features
        processed_features = process_input(car_features)
        
        #Make prediction
        prediction = model.predict(processed_features)
        
        #Return prediction
        return {"predicted_price": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

#API check endpoint to verify model availability
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

#Run the FastAPI server when executed as a script
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)