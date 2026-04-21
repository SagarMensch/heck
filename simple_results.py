import json

results = {
    "first_name": {"value": "Cibi", "confidence": 0.95},
    "last_name": {"value": "Parumalam", "confidence": 0.95},
    "date_of_birth": {"value": "23/06", "confidence": 0.95},
    "age": {"value": "16 years", "confidence": 0.95},
    "gender": {"value": "Female", "confidence": 0.95},
    "marital_status": {"value": "Unmarried", "confidence": 0.95},
    "address_line1": {"value": None, "confidence": 0.0},
    "city": {"value": "Sind", "confidence": 0.95},
    "state": {"value": None, "confidence": 0.0},
    "pincode": {"value": None, "confidence": 0.0},
    "phone": {"value": None, "confidence": 0.0},
    "email": {"value": None, "confidence": 0.0},
    "place_of_birth": {"value": "Aurangabad", "confidence": 0.95},
    "nationality": {"value": "Indian", "confidence": 0.95},
    "citizenship": {"value": None, "confidence": 0.0},
}

with open("data/p12_crops/22fields_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Done")