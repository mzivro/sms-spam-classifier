from src.schemas import TextMessage, PredictionResponse
from fastapi import APIRouter, Request

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
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
    model = request.app.state.model_service

    prediction = model.predict(data.text)

    return PredictionResponse(prediction=prediction)
