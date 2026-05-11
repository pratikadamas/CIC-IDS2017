
import pandas as pd
import asyncio
# pyrefly: ignore [missing-import]
import joblib

from datetime import datetime

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

# ==========================================
# FASTAPI
# ==========================================

app = FastAPI()

templates = Jinja2Templates(
    directory="templates"
)

# ==========================================
# CORS
# ==========================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "ids_multiclass_pipeline1.pkl"
)

# ==========================================
# LOAD ENCODER
# ==========================================

encoder = joblib.load(
    "label_encoder1.pkl"
)

# ==========================================
# LOAD FEATURES
# ==========================================

feature_columns = joblib.load(
    "feature_columns1.pkl"
)

# ==========================================
# LOAD LIVE TRAFFIC
# ==========================================

traffic_data = pd.read_csv(
    "live_attack.csv"
)

# ==========================================
# HOME
# ==========================================

@app.get(
    "/",
    response_class=HTMLResponse
)

async def dashboard(
    request: Request
):

    return templates.TemplateResponse(

        "indexx.html",

        {
            "request": request
        }
    )

# ==========================================
# WEBSOCKET
# ==========================================

@app.websocket("/ws")

async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()

    print("WebSocket Connected")

    try:

        while True:

            for _, row in traffic_data.iterrows():

                sample = pd.DataFrame([row])

                # remove target
                X_sample = sample.drop(
                    columns=["Attack Type"],
                    errors="ignore"
                )

                # clean names
                X_sample.columns = (
                    X_sample.columns.str.strip()
                )

                # align features
                X_sample = X_sample[
                    feature_columns
                ]

                # ==========================
                # PREDICTION
                # ==========================

                prediction = model.predict(
                    X_sample
                )

                probabilities = (
                    model.predict_proba(
                        X_sample
                    )
                )

                # decode class
                predicted_class = (
                    encoder.inverse_transform(
                        prediction
                    )[0]
                )

                # highest confidence
                confidence = max(
                    probabilities[0]
                )

                # ==========================
                # SEVERITY
                # ==========================

                if predicted_class == "BENIGN":

                    severity = "LOW"

                elif confidence >= 0.90:

                    severity = "HIGH"

                elif confidence >= 0.70:

                    severity = "MEDIUM"

                else:

                    severity = "LOW"

                # ==========================
                # RESPONSE
                # ==========================

                response = {

                    "timestamp":
                    str(datetime.now()),

                    "prediction":
                    predicted_class,

                    "confidence":
                    round(
                        confidence * 100,
                        2
                    ),

                    "severity":
                    severity

                }

                print(response)

                await websocket.send_json(
                    response
                )

                await asyncio.sleep(1)

    except Exception as e:

        import traceback

        traceback.print_exc()

