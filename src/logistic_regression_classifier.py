
# system tools
import os
# data munging tools
import pandas as pd
# importing functions from preprocessing script 
import sys
sys.path.append("utils")
# import my functions in the utils folder 
from preprocessingutils import load_data
from preprocessingutils import train_test
from preprocessingutils import vectorize
# Machine learning stuff
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
# Visualisation
import matplotlib.pyplot as plt
# saving models 
from joblib import dump



def logistic_classifier(X_train_feats, y_train, X_test_feats, y_test):
    # trained on the training data - logistic regression
    classifier = LogisticRegression(random_state=42)
    classifier.fit(X_train_feats, y_train)
    # get predictions 
    y_pred = classifier.predict(X_test_feats)
    # show metrics for how model is performing 
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    # save the report 
    with open(os.path.join("out", "logistic_classification_report.txt"), "w") as f:
        f.write(classifier_metrics)
    # save the model
    dump(classifier, os.path.join("models", "LogisticClassifier.joblib"))
    return classifier, y_pred

def main():
    # load the data
    X, y = load_data()
    # make the train and test split in the data
    X_train, X_test, y_train, y_test = train_test(X, y)
    # vectorize the data 
    vectorizer, X_train_feats, X_test_feats = vectorize(X_train, X_test)
    # run logistic regression classifier 
    classifier, y_pred = logistic_classifier(X_train_feats, y_train, X_test_feats, y_test)


if __name__=="__main__":
    main()

  
    














