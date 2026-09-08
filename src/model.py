from huggingface_hub import hf_hub_download

import src.utils as utils
import pickle


class Model:
    """
    Class responsible for loading and serving the spam classification model.

    If the serialized model file does not exist, it will be downloaded from hugging face.
    """

    def __init__(self):
        """
        Loads model from cache, if it does not exist, downloads it
        from Hugging Face.
        """

        path = hf_hub_download(
            repo_id="mzivro/my-models",
            filename="sms_spam_classifier.pkl",
            repo_type="model",
        )

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
