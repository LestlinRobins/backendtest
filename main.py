from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import pandas as pd
from rf_2_pred import CompletePhasePredictionSystem  # Import your class
import os
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
predictor = CompletePhasePredictionSystem()

# Load model at startup (replace with your actual model path)
predictor.load_model("phase_identifier_20250716_101849.joblib")

@app.post("/process-csv")
async def process_csv(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    input_path = "test_data.csv"
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Process with your prediction system
    output_path = "phase_predictions.csv"
    result = predictor.predict_complete(input_path, output_path)
    
    if result is None:
        return {"error": "Prediction failed"}
    
    # Return the output CSV
    return FileResponse(
        output_path,
        media_type="text/csv",
        filename="phase_predictions.csv"
    )
