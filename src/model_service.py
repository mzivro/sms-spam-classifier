import src.utils as utils
import pickle


class ModelService:
    """
    Service responsible for loading and serving the spam classification model.

    If the serialized model file does not exist, a new model is trained and
    saved automatically.

    Parameters
    ----------
    path : str
        Path to the serialized model file.
    """
    def __init__(self, path: str):
        """
        Load the trained model from disk or train a new one if necessary.

        Parameters
        ----------
        path : str
            Path to the serialized model file.
        """
        try:
            with open(path, "rb") as file:
                self.model = pickle.load(file)
        except FileNotFoundError:
            print("Model file not found, beginning training")
            utils.train(path)

            with open(path, "rb") as file:
                self.model = pickle.load(file)

    def predict(self, text: str) -> int:
        """
        Predict whether a text message is spam.

        The input text is preprocessed before being passed to the trained model.

        Parameters
        ----------
        text : str
            Input text message.

        Returns
        -------
        int
            Predicted class label where 0 indicates ham and 1 indicates spam.
        """
        preprocessed_text = utils.preprocess(text)

        return int(self.model.predict([preprocessed_text])[0])
