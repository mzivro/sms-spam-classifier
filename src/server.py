from src.schemas import TextMessage, PredictionResponse
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from src.model import Model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the application lifecycle.

    The machine learning model is loaded during application startup and stored
    in the application state for reuse across requests.

    Parameters
    ----------
    app : FastAPI
        FastAPI application instance.

    Yields
    ------
    None
        Control back to the application after initialization.
    """
    app.state.model = Model()

    yield

    del app.state.model


app = FastAPI(title="SMS Spam Classifier API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """
    Return the health status of the API service.

    Returns
    -------
    dict
        Dictionary containing the service status and the device used
        for inference.

        - "status" : str
            Always "ok" when the endpoint is reachable.
    """
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """
    Check whether the spam classification model is loaded.

    Returns
    -------
    dict
        Dictionary containing status="ready" when the model is
        available.

    Raises
    ------
    fastapi.HTTPException
        Raised with HTTP status code 503 when the model has not been
        loaded yet.
    """
    if not hasattr(app.state, "model"):
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    return {
        "status": "ready",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: Request, data: TextMessage) -> PredictionResponse:
    """
    Predict whether a text message is spam or ham.

    Parameters
    ----------
    request : Request
        FastAPI request object containing the application state.
    data : TextMessage
        Request body containing the input text message.

    Returns
    -------
    PredictionResponse
        Prediction result where 0 represents ham and 1 represents spam.
    """
    model = request.app.state.model

    prediction = model.predict(data.text)

    return PredictionResponse(prediction=prediction)
