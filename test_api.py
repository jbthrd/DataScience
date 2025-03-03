import json
import requests

# URL for running the API locally 
BASE_URL = "http://localhost:8000"

def test_api():
    # Health check request
    health_response = requests.get(f"{BASE_URL}/health")
    print("Health Check Response:", health_response.json())
    
    # Sample car data for prediction
    sample_car = {
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
    
    # Prediction request
    prediction_response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(sample_car)
    )
    try:
        print("Prediction Response:", prediction_response.json())
    except Exception as e:
        print("Error decoding prediction response:", e)

if __name__ == "__main__":
    test_api()
