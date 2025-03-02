import requests
import json

# Local testing URL (when running the API locally)
BASE_URL = "http://localhost:8000"

# For Render deployment, replace with your actual Render URL
# BASE_URL = "https://car-price-prediction-api.onrender.com"

def test_health_endpoint():
    """Test the health check endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check Response:", response.json())
    assert response.status_code == 200

def test_prediction_endpoint():
    """Test the prediction endpoint with sample data"""
    # Sample car data matching the expected input format
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
    
    # Make the POST request to the prediction endpoint
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(sample_car)
    )
    
    # Print the full response for debugging
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {response.json()}")
    except:
        print(f"Raw Response: {response.text}")
    
    # Assertions to verify the response
    assert response.status_code == 200
    assert "predicted_price" in response.json()
    
    # The predicted price should be a positive number
    predicted_price = response.json()["predicted_price"]
    assert isinstance(predicted_price, (int, float))
    assert predicted_price > 0
    
    print(f"Predicted Price: ${predicted_price:.2f}")

def test_multiple_cars():
    """Test prediction with multiple car configurations"""
    test_cases = [
        {
            "description": "Luxury car with high specs",
            "data": {
                "Manufacturer": "Mercedes-Benz",
                "Model": "S-Class",
                "Category": "Sedan",
                "Leather_interior": "yes",
                "Mileage": "15000 km",
                "Cylinders": 8,
                "Engine_volume": "4.0 Turbo",
                "Doors": "4",
                "Wheel": "Left wheel",
                "Color": "Silver",
                "Airbags": 12,
                "Prod_year": 2020,
                "Drive_wheels": "4wd",
                "Gear_box_type": "automatic",
                "Fuel_type": "gasoline"
            }
        },
        {
            "description": "Economy car with basic specs",
            "data": {
                "Manufacturer": "Honda",
                "Model": "Civic",
                "Category": "Hatchback",
                "Leather_interior": "no",
                "Mileage": "45000 km",
                "Cylinders": 4,
                "Engine_volume": "1.5",
                "Doors": "4",
                "Wheel": "Left wheel",
                "Color": "Blue",
                "Airbags": 4,
                "Prod_year": 2016,
                "Drive_wheels": "front",
                "Gear_box_type": "manual",
                "Fuel_type": "gasoline"
            }
        }
    ]
    
    for case in test_cases:
        print(f"\nTesting: {case['description']}")
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(case['data'])
        )
        
        # Print results
        if response.status_code == 200:
            price = response.json()["predicted_price"]
            print(f"Predicted Price: ${price:.2f}")
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    print("Testing Car Price Prediction API...")
    
    # Run the tests
    try:
        test_health_endpoint()
        print("✓ Health check endpoint is working")
    except Exception as e:
        print(f"✗ Health check test failed: {str(e)}")
    
    try:
        test_prediction_endpoint()
        print("✓ Prediction endpoint is working")
    except Exception as e:
        print(f"✗ Prediction test failed: {str(e)}")
    
    try:
        test_multiple_cars()
        print("✓ Multiple car predictions test completed")
    except Exception as e:
        print(f"✗ Multiple car test failed: {str(e)}")