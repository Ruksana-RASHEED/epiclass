""" Tests the functionality of the epileptic seizure classifiers and api
"""

import unittest
import os
import pandas as pd
from pandas.util.testing import assert_frame_equal
import joblib
import json

from api import app


MODEL_DIR = 'models'
TEST_DIR = 'test_cases'


def read_features_targets(filename):
    """Read features and targets from a csv file

    Args:
        filename: str
            Path to the csv file to read

    Returns:
        A tuple features, targets
            features: pandas DataFrame
                features in the file
            targets: pandas Series
                targets in the file
    """
    all_data = pd.read_csv(filename, index_col=0)
    features = all_data.drop('y', axis=1)
    targets = all_data['y']
    return features, targets


def read_features_targets_2c(filename):
    """Read features and targets from a csv file to use in binary classification

        Args:
            filename: str
                Path to the csv file to read

        Returns:
            A tuple features, targets
                features: pandas DataFrame
                    features in the file
                targets: pandas Series
                    targets in the file, as 0 or 1, where 1 means the
                    classification was 1 and 0 is any other class
        """
    features, target_class = read_features_targets(filename)
    targets = (target_class == 1).astype(int)
    return features, targets


class TestTwoClassPcaSvm(unittest.TestCase):
    """Tests for the two-class PCA SVM pipeline
    """

    def setUp(self):
        """Load the two-class PCA SVM pipeline
        """
        filename = os.path.join(MODEL_DIR, 'two_class_pca_svm.z')
        self.model = joblib.load(filename)

    def test_predict(self):
        """Test that the prediction works for one test case
        """
        _, class_name, method_name = self.id().split('.')
        directory = os.path.join(TEST_DIR, class_name, method_name)
        filename = os.path.join(directory, 'test_data.csv')
        x_test, y_test = read_features_targets(filename)
        y_pred = self.model.predict(x_test)
        self.assertEqual(y_pred[0], (y_test==1).astype(int).iloc[0])

    def test_confusion_matrix(self):
        """Test that the test data confusion matrix has not changed
        """
        _, class_name, method_name = self.id().split('.')
        directory = os.path.join(TEST_DIR, class_name, method_name)
        filename = os.path.join(directory, 'test_data.csv')
        x_test, y_test = read_features_targets_2c(filename)
        y_pred = self.model.predict(x_test)
        confusion = pd.crosstab(y_test, y_pred)
        expected_filename = os.path.join(directory, 'expected_confusion.csv')
        expected_result = pd.read_csv(expected_filename, index_col=0)
        expected_result.columns = confusion.columns
        assert_frame_equal(confusion, expected_result)


class TestFiveClassPcaSvm(unittest.TestCase):

    def setUp(self):
        """Load the five-class PCA SVM pipeline
        """
        filename = os.path.join(MODEL_DIR, 'five_class_pca_svm.z')
        self.model = joblib.load(filename)

    def test_predict(self):
        """Test that the prediction works for one test case
        """
        _, class_name, method_name = self.id().split('.')
        directory = os.path.join(TEST_DIR, class_name, method_name)
        filename = os.path.join(directory, 'test_data.csv')
        x_test, y_test = read_features_targets(filename)
        y_pred = self.model.predict(x_test)
        self.assertEqual(y_pred[0], y_test.iloc[0])

    def test_confusion_matrix(self):
        """Test that the confusion matrix has not changed
        """
        _, class_name, method_name = self.id().split('.')
        directory = os.path.join(TEST_DIR, class_name, method_name)
        filename = os.path.join(directory, 'test_data.csv')
        x_test, y_test = read_features_targets(filename)
        y_pred = self.model.predict(x_test)
        confusion = pd.crosstab(y_test, y_pred)
        expected_filename = os.path.join(directory, 'expected_confusion.csv')
        confusion.to_csv(expected_filename)
        expected_result = pd.read_csv(expected_filename, index_col=0)
        expected_result.columns = confusion.columns
        assert_frame_equal(confusion, expected_result)


class TestApi(unittest.TestCase):

    def setUp(self):
        """Start the api running
        """
        app.config['TESTING'] = True
        self.app = app.test_client()

    def create_get_request(self, features):
        """Make a get request to send to the API

        Args:
            features: pandas Series
                feature values
        Returns: str
            Get request for API
        """
        query = features.to_json()
        return '/?query=' + query

    def test_classify(self):
        """Test that the API returns the correct classification
        """
        _, class_name, method_name = self.id().split('.')
        directory = os.path.join(TEST_DIR, class_name, method_name)
        filename = os.path.join(directory, 'test_data.csv')
        x_test, y_test = read_features_targets(filename)
        get_request = self.create_get_request(2047.0 * x_test)
        response = self.app.get(get_request)
        prediction = json.loads(response.data)['prediction']
        self.assertEqual(prediction, 'Not seizure')

if __name__ == '__main__':
    unittest.main()
