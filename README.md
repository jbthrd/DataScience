# Car Price Prediction API

This is our FastAPI application that predicts car prices based on vehicle features using an XGBoost model.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API locally
uvicorn main:app --reload

# Test the API
python test_api.py
```

The API will be available at http://localhost:8000 with interactive documentation at http://localhost:8000/docs.

## API Usage

### Prediction Endpoint

```
POST /predict
```

### Expected Input Format

The API expects a JSON object with the following properties:

| Field            | Type    | Description                                               | Example      |
| ---------------- | ------- | --------------------------------------------------------- | ------------ |
| Manufacturer     | string  | Car manufacturer                                          | "Toyota"     |
| Model            | string  | Car model                                                 | "Camry"      |
| Category         | string  | Vehicle category                                          | "Sedan"      |
| Leather_interior | string  | "yes" or "no"                                             | "yes"        |
| Mileage          | string  | Distance with unit                                        | "35000 km"   |
| Cylinders        | integer | Number of cylinders                                       | 4            |
| Engine_volume    | string  | Engine size with optional "Turbo"                         | "2.5 Turbo"  |
| Doors            | string  | Number of doors (can be "4", "02-Mar", "04-May", or ">5") | "4"          |
| Wheel            | string  | Steering wheel position                                   | "Left wheel" |
| Color            | string  | Car color                                                 | "Black"      |
| Airbags          | integer | Number of airbags                                         | 6            |
| Prod_year        | integer | Production year                                           | 2018         |
| Drive_wheels     | string  | Drivetrain type ("front", "rear", or "4wd")               | "front"      |
| Gear_box_type    | string  | Transmission type                                         | "automatic"  |
| Fuel_type        | string  | Fuel type                                                 | "gasoline"   |

### Sample Request

```json
{
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
```

### Sample Response

```json
{
  "predicted_price": 23450.75
}
```
