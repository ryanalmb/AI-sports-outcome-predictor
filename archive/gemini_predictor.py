from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict_match(match_data: dict):
    # This is where the Gemini analysis will happen.
    # For now, it just returns the data it received.
    return {"prediction": "coming soon", "received_data": match_data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)