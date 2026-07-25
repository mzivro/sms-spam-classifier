from pydantic import BaseModel


class TextMessage(BaseModel):
    """
    Request schema containing a text message.

    Attributes
    ----------
    text : str
        Text message to classify.
    """
    text: str


class PredictionResponse(BaseModel):
    """
    Response schema containing the predicted class.

    Attributes
    ----------
    prediction : int
        Predicted label where 0 represents ham and 1 represents spam.
    """
    prediction: int
