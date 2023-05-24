# system tools
import os
# data munging tools - having an issue here even after running the setup.sh script 
import pandas as pd
# Machine learning stuff
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
# saving models 
from joblib import dump

def load_data(): 
    filename = os.path.join("in", "fake_or_real_news.csv")
    data = pd.read_csv(filename, index_col=0)
    X = data["text"]
    y  = data["label"]
    return X, y
    

def train_test(X, y):
    # making the train/test split 
    X_train, X_test, y_train, y_test = train_test_split(X,              
                                                        y,               
                                                        test_size=0.2,   
                                                        random_state=42) 
    return X_train, X_test, y_train, y_test


def vectorize(X_train, X_test):
    # creating vectorizer object 
    vectorizer = TfidfVectorizer(ngram_range = (1,2),    
                                lowercase =  True,       
                                max_df = 0.95,            
                                min_df = 0.05,           
                                max_features = 500)     
    # fit the vectorizer to the training data  
    X_train_feats = vectorizer.fit_transform(X_train)
    # transform test data 
    X_test_feats = vectorizer.transform(X_test)
    # save the vectorizer 
    dump(vectorizer, os.path.join("models", "Vectorizer.joblib"))
    return vectorizer, X_train_feats, X_test_feats

