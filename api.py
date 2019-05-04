"""API for accessing 2-parameter classification of epileptic seizure data
"""
import os

import joblib
import pandas as pd
from flask import Flask
from flask_restful import reqparse, Api, Resource

app = Flask(__name__)
api = Api(app)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictSeizure(Resource):
    """Predict seizure using PCA-SVM model
    """
    SCALE_FACTOR = 2047.0
    MODEL_PATH = os.path.join('models', 'two_class_pca_svm.z')

    def __init__(self):
        """Load the prediction model
        """
        self.model = self.load_model()

    def get(self):
        """Handle a GET request for classifying brain activity data

        Returns: str
            JSON-formatted dict with one entry, 'prediction', which is either
            'Seizure' or 'Not Seizure'
        """
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        # convert query to array
        model_input = self.convert_query(user_query)
        prediction = self.model.predict(model_input / self.SCALE_FACTOR)
        # Output either 'Seizure' or 'Not Seizure' along with the score
        print(prediction)
        if prediction == 0:
            pred_text = 'Not seizure'
        else:
            pred_text = 'Seizure'
        # create JSON object
        output = {'prediction': pred_text}
        return output

    def load_model(self):
        return joblib.load(self.MODEL_PATH)

    def convert_query(self, query):
        return pd.read_json(query) / self.SCALE_FACTOR


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSeizure, '/')

if __name__ == '__main__':
    app.run(debug=True)
