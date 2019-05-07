"""API for accessing 2-parameter classification of epileptic seizure data
"""
import os
import textwrap

import joblib
import pandas as pd
from flask import Flask, render_template_string, make_response
from flask_restful import reqparse, Api, Resource

app = Flask(__name__)
api = Api(app)


class PredictSeizure(Resource):
    """Predict seizure using PCA-SVM model
    """
    SCALE_FACTOR = 2047.0
    MODEL_PATH = os.path.join('models', 'two_class_pca_svm.z')

    def __init__(self):
        """Constructor for PredictSeizure Resource
        """
        self.model = self.load_model()
        self.form_html = self.create_form_html()

    def load_model(self):
        """Load the prediction model
        """
        return joblib.load(self.MODEL_PATH)

    @staticmethod
    def create_form_html():
        """Create the html to display an input form
        """
        data_file = os.path.join('data', 'data.csv')
        data = pd.read_csv(data_file, index_col=0)
        example1 = data.iloc[0, :178]
        example2 = data.iloc[4340, : 178]
        placeholder = ', '.join(example1.astype(str))
        example_str1 = textwrap.fill(placeholder, 80)
        example_str2 = textwrap.fill(', '.join(example2.astype(str)), 80)
        form_html = ('''
            <html><body>
                <h1>Binary classifier for Epileptic Seizure Recognition Data 
                Set</h1>
                <h2>Please enter features for classification</h1>
                (178 integers, separated by commas)
                <form method="post" action="">
                    <textarea name="query" cols="80" rows="10">'''
                     + placeholder
                     + ''' </textarea>
                    <input type="submit">
                </form>
                <p> Example non-seizure data point:
                '''
                     + example_str1
                     + '''<p> Example seizure data point: '''
                     + example_str2
                     + '''</body></html>''')
        return form_html

    def get(self):
        """Handle a GET request

        Returns: flask.wrappers.Response
            A response that requests rendering of a form that the user can
            paste the data into
        """
        response = make_response(render_template_string(self.form_html), 200)
        response.headers['mime-type'] = 'text/html'
        return response

    def post(self):
        """Handle a POST request for classifying brain activity data

        Returns: str
            JSON-formatted dict with one entry, 'prediction', which is either
            'Seizure' or 'Not Seizure'
        """
        # argument parsing
        parser = reqparse.RequestParser()
        parser.add_argument('query', 'str')
        args = parser.parse_args()
        user_query = args['query']
        # convert query to array
        model_input = self.convert_query(user_query)
        prediction = self.model.predict(model_input)[0]
        response = make_response(render_template_string(
            self.make_post_return(prediction)), 200)
        response.headers['mime-type'] = 'text/html'
        return response

    @staticmethod
    def make_post_return(prediction):
        if prediction == 0:
            pred_text = 'Model predicts that it is not a seizure'
        else:
            pred_text = 'Model predicts that it is a seizure'
        html = ('''<html><body>
                <h1>Binary classifier for Epileptic Seizure Recognition Data 
                Set</h1>
                <h2>'''
                + pred_text
                + '''</body></html>''')
        return html

    def convert_query(self, query):
        """Convert a POST query into a format used by the prediction model

        Args:
            query: str
                Comma-separated list of 178 integers. Features for use in the
                prediction model
        Returns: numpy ndarray
            Input into scikit-learn prediction model
        """
        rescaled = pd.read_json('[' + query + ']') / self.SCALE_FACTOR
        model_input = rescaled.values.reshape(1, -1)
        return model_input


if __name__ == '__main__':
    # Route the URL to the resource
    api.add_resource(PredictSeizure, '/')
    app.run(debug=True)
