# system tools
import os
# data munging tools
import pandas as pd
# Machine learning stuff
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
# saving models 
from joblib import dump

def load_data(): 
    filename = os.path.join("in", "fake_or_real_news.csv")
    data = pd.read_csv(filename, index_col=0)
    # these column names will be specific to whatever dataset is loaded in 
    X = data["text"]
    y  = data["label"]
    return X, y


def train_test(X, y):
    global X_train
    global X_test
    global y_train
    global y_test
    # making the train/test split 
    X_train, X_test, y_train, y_test = train_test_split(X,               # texts for the model
                                                        y,               # classification labels
                                                        test_size=0.2,   # create an 80/20 split (20% test data, 80% train data)
                                                        random_state=42) # random state for reproducibility - like set.seed() in R
    return X_train, X_test, y_train, y_test

def vectorize(X_train, X_test):
    global vectorizer
    global X_train_feats
    global X_test_feats
    # creating vectorizer object - which makes the text features into numerical vectors and makes it easier/faster to work with  
    vectorizer = TfidfVectorizer(ngram_range = (1,2),    # unigrams and bigrams, but can be more 
                                lowercase =  True,       # making everything lowercase 
                                max_df = 0.95,           # removes words in more than 95% of documents, the super common ones 
                                min_df = 0.05,           # removes words in less than 5% of documents, the super rare ones and possibly misspelled words 
                                max_features = 500)      # keep only top 500 features
    # fit the vectorizer to the training data - calculates the mean and variance of each feature 
    # then transforms all the features with the respective mean and variance to scale the training data 
    X_train_feats = vectorizer.fit_transform(X_train)
    # transform test data - using the same mean and variance calculated from the training data to transform the test data 
    # do not want to also fit the test data or the model will learn from that too and will not be an accurate indicator of how the model performs 
    X_test_feats = vectorizer.transform(X_test)
    # save the vectorizer 
    dump(vectorizer, os.path.join("models", "Vectorizer.joblib"))
    return vectorizer, X_train_feats, X_test_feats


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
    # above here, should I get rid of the classifier and y_pred since I don't need them?
    

if __name__=="__main__":
    main()
