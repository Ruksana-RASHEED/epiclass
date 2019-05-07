#!/usr/bin/env python3

# import modules used here -- sys is a very standard one
import sys, argparse, logging
from epiclass import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Demonstration of classification on the Epileptic Seizure"
                    " Recognition Data Set")
    help_text = """Action to take. Choose one or more of
                      explore - make several plots of the data, including of the
                          Principal Component Analysis (PCA) transformation of
                          the features
                      pca_svm2 - train a binary classifier (seizure vs
                          non-seizure) using a pipeline of PCA and a support 
                          vector machine
                      pca_svm5 - train a multiclass classifier using a pipeline 
                          of PCA and a support vector machine
                      rf - train a multiclass classifier using a random decision 
                           forest
                      nn - train a multiclass classifier using an artificial 
                           neural network
                  All the training methods save the models to the model 
                  directory and additionally save a confusion matrix to the 
                  outputs directory.
                  """
    parser.add_argument('action', nargs='+',
                        choices=['explore', 'pca_svm2', 'pca_svm5', 'rf', 'nn'],
                        help=help_text)
    args = parser.parse_args()
    run(args.action)


