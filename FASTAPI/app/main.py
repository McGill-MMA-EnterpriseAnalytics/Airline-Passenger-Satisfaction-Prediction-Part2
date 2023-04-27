from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from pathlib import Path


# Define the FastAPI app
app = FastAPI()

# Define the input schema - need to rename column names in the model code
class InputData(BaseModel):
    Class: int
    Flight_Distance: float                   
    Inflight_wifi_service:int              
    Departure_Arrival_time_convenient: int  
    Ease_of_Online_booking: int             
    Online_boarding: int                    
    Seat_comfort: int                       
    Inflight_entertainment: int          
    On_board_service: int                   
    Leg_room_service: int                
    Baggage_handling: int                   
    Checkin_service: int                   
    Inflight_service: int                 
    Cleanliness: int                      
    Departure_Delay_in_Minutes: float         
    Customer_Type_Loyal_Customer: int
    Type_of_Travel_Business_travel: int     


# Define the filepath for the pickle file
__version__ = "v1.0.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


# Load the trained pipeline from the pickle file
with open(f"{BASE_DIR}/trained_pipeline_{__version__}.pkl", "rb") as f:
    pipeline = pickle.load(f)


@app.get("/")
def read_root():
    return {"Hello": "World"}

# Define the predict endpoint
@app.post("/predict")
async def predict(data: InputData):
    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame(data=[data.dict()])

    # Use the trained pipeline to make predictions
    prediction = pipeline.predict(input_df)

    # Return the prediction as a dictionary
    return {'prediction': prediction[0]}

