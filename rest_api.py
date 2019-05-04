import os

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pandas as pd
import joblib

app = Flask(__name__)
api = Api(app)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictSeizure(Resource):

    SCALE_FACTOR = 2047.0
    MODEL_PATH = os.path.join('models', 'two_class_pca_svm.z')

    def __init__(self):
        self.model = self.load_model()

    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        # convert query to array
        model_input = self.convert_query(user_query)
        prediction = self.model.predict(model_input)
        pred_proba = self.model.predict_proba(model_input)
        # Output either 'Seizure' or 'Not Seizure' along with the score
        if prediction == 0:
            pred_text = 'Not seizure'
        else:
            pred_text = 'Seizure'
        # round the predict proba value and set to new variable
        confidence = round(pred_proba[0], 3)
        # create JSON object
        output = {'prediction': pred_text, 'confidence': confidence}
        return output

    def load_model(self):
        return joblib.load(self.MODEL_PATH)

    def convert_query(self, query):
        return pd.read_json(query)/self.SCALE_FACTOR


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSeizure, '/')

if __name__ == '__main__':
    app.run(debug=True)