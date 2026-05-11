import pandas as pd
import asyncio
# pyrefly: ignore [missing-import]
import joblib
from datetime import datetime

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

# =========================
# FASTAPI APP
# =========================

app = FastAPI()

# =========================
# TEMPLATE CONFIG
# =========================

templates = Jinja2Templates(directory="templates")

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODEL
# =========================

model = joblib.load(
    "ids_pipeline.pkl"
)

# =========================
# LOAD FEATURE COLUMNS
# =========================

feature_columns = joblib.load(
    "feature_columns.pkl"
)

# =========================
# LOAD TRAFFIC DATA
# =========================

traffic_data = pd.read_csv(
    "live_attack.csv"
)

# =========================
# DASHBOARD ROUTE
# =========================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def dashboard(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index2.html"
    )

# =========================
# PREDICTION API
# =========================
@app.post("/predict")
def predict(data: dict):

    try:

        # convert incoming json to dataframe
        df = pd.DataFrame([data])

        # ensure exact feature order
        df = df[feature_columns]

        # get probabilities
        probability = model.predict_proba(df)

        # probability of ATTACK class
        attack_probability = float(
            probability[0][1]
        )

        # =========================
        # CUSTOM IDS THRESHOLD
        # =========================

        THRESHOLD = 0.30

        # prediction logic
        if attack_probability >= THRESHOLD:

            prediction = 1

        else:

            prediction = 0

        # =========================
        # CONFIDENCE LOGIC
        # =========================

        if prediction == 1:

            confidence = attack_probability

        else:

            confidence = 1 - attack_probability

        # =========================
        # SEVERITY LOGIC
        # =========================

        if attack_probability >= 0.80:

            severity = "HIGH"

        elif attack_probability >= 0.40:

            severity = "MEDIUM"

        else:

            severity = "LOW"

        return {

            "prediction": prediction,

            "label":
            "ATTACK"
            if prediction == 1
            else "BENIGN",

            "confidence": round(
                confidence * 100,
                2
            ),

            "attack_probability": round(
                attack_probability * 100,
                2
            ),

            "severity": severity

        }

    except Exception as e:

        return {
            "error": str(e)
        }

# =========================
# WEBSOCKET STREAM
# =========================

# =========================
# WEBSOCKET STREAM
# =========================

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

                # remove unused columns
                X_sample = sample.drop(
                    columns=["Destination Port"],
                    errors="ignore"
                )

                # clean columns
                X_sample.columns = (
                    X_sample.columns.str.strip()
                )

                # align features
                X_sample = X_sample[
                    feature_columns
                ]

                # =========================
                # PROBABILITY
                # =========================

                probability = model.predict_proba(
                    X_sample
                )

                attack_probability = float(
                    probability[0][1]
                )

                # =========================
                # CUSTOM THRESHOLD
                # =========================

                THRESHOLD = 0.30

                if attack_probability >= THRESHOLD:

                    prediction = 1

                else:

                    prediction = 0

                # =========================
                # CONFIDENCE
                # =========================

                if prediction == 1:

                    confidence = attack_probability

                else:

                    confidence = 1 - attack_probability

                # =========================
                # SEVERITY
                # =========================

                if attack_probability >= 0.80:

                    severity = "HIGH"

                elif attack_probability >= 0.40:

                    severity = "MEDIUM"

                else:

                    severity = "LOW"

                # =========================
                # RESPONSE
                # =========================

                response = {

                    "timestamp":
                    str(datetime.now()),

                    "prediction":
                    prediction,

                    "label":
                    "ATTACK"
                    if prediction == 1
                    else "BENIGN",

                    "confidence":
                    round(confidence * 100, 2),

                    "attack_probability":
                    round(attack_probability * 100, 2),

                    "severity":
                    severity

                }

                print(response)

                await websocket.send_json(
                    response
                )

                await asyncio.sleep(2)

    except Exception as e:

        import traceback

        traceback.print_exc()
async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()

    print("WebSocket Connected")

    try:

        while True:

            for _, row in traffic_data.iterrows():

                sample = pd.DataFrame([row])

                X_sample = sample.drop(
                    columns=["Destination Port","Label"],
                    errors="ignore"
                )

                X_sample.columns = (
                    X_sample.columns.str.strip()
                )

                X_sample = X_sample[
                    feature_columns
                ]

                prediction = model.predict(
                    X_sample
                )

                probability = model.predict_proba(
                    X_sample
                )

                response = X_sample.to_dict(
                    orient="records"
                )[0]

                response["prediction"] = int(
                    prediction[0]
                )

                response["confidence"] = float(
                    probability[0][1]
                )

                response["timestamp"] = str(
                    datetime.now()
                )

                print(response)

                await websocket.send_json(
                    response
                )

                await asyncio.sleep(1)

    except Exception as e:

        import traceback

        traceback.print_exc()