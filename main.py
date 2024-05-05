from fastapi import FastAPI
from predict_spend import predictSpend
import pandas as pd

app = FastAPI()

@app.get("/predict-spend")
async def result_model():
    data = pd.read_csv("spends_data.csv") 
    params = {
    'country_code':'US',
    'age':31,
    'aadhar_gender': 'M',
    'occupation':'Salaried',
    'pincode':600116,
    'annual_income':2000000,
    'marital_status':'Unmarried',
    'networth':5000000,
    'source_of_funds':'Salary',
   }

    predict_spend = predictSpend(data, params)      
    return predict_spend

@app.get("/")
def read_root():
    return {"message": "Welcome to the IMS-Backend API!"}
