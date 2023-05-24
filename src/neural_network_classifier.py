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
from sklearn.neural_network import MLPClassifier
# saving models 
from joblib import dump



def neural_net_classifier(X_train_feats, y_train, X_test_feats, y_test): 
    # trained on the training data - neural network
    classifier = MLPClassifier(activation = "logistic", # function for the hidden layer, logistic = sigmoid function 
                            hidden_layer_sizes = (22,), # number chosen, in this case 22 means 22 neurons in the 22 hidden layers
                            max_iter=1000, # max number of iterations until model converges, may be before it reaches this number 
                            random_state = 42) 
    # classifier is fit to the training data to learn 
    classifier.fit(X_train_feats, y_train)
    # get predictions - classifier is now used to get predictions from the test data  
    y_pred = classifier.predict(X_test_feats)
    # show metrics for how model is performing 
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    # save the report 
    with open(os.path.join("out", "neural_net_classification_report.txt"), "w") as f:
        f.write(classifier_metrics)
    # save the model 
    dump(classifier, os.path.join("models", "NNclassifier.joblib"))
    return classifier, y_pred


def main():
    # load the data
    X, y = load_data()
    # make the train and test split in the data
    X_train, X_test, y_train, y_test = train_test(X, y)
    # vectorize the data 
    vectorizer, X_train_feats, X_test_feats = vectorize(X_train, X_test)
    # run neural net classifier 
    classifier, y_pred = neural_net_classifier(X_train_feats, y_train, X_test_feats, y_test)
    

    
if __name__=="__main__":
    main()
