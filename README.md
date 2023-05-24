
# Assignment 2 - Text classification benchmarks

## Github repo link 

This assignment can be found at my github [repo](https://github.com/ameerwald/cds_lang_exam_assignment2)
 
## The data

A dataset of fake and real news headlines and can be found [here] (https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news). It contains the headline, text of the story and label (fake or real). 

## Assignment description

This assignment asked for two different scripts training a logistic regression classifier and a neural network classifer. This was to be done using ```scikit-learn```. Both scripts required three things:

- Be executed from the command line
- Save the classification report to the folder called ```out```
- Save the trained models and vectorizers to the folder called ```models```

## Repository 

| Folder         | Description          
| ------------- |:-------------:
| In      | Data - Fake or real new corpus
| Models  | Saved models for the vectorizer, logistic regression and neural network classifiers 
| Notes | Jupyter notebook notes and old .py scripts      
| Out  | Classification Reports, one for each classifier    
| Src  | Two py scrips one for each classifier, labeled accordingly  
| Utils  | Utilities I created       


## To run the scripts 

1. Clone the repository, either on ucloud or something like worker2
2. From the command line, at the /cds_vis_exam_assignment1/ folder level, run the following lines of code. 

This will create a virtual environment, install the correct requirements.
``` 
bash setup.sh
```
While this will run the scripts and deactivate the virtual environment when it is done. 
```
bash run.sh
```

This has been tested on an ubuntu system on ucloud and therefore could have issues when run another way.

## Discussion of Results
When comparing the classification reports it appears that the classifiers are almost the same in terms of performance. Both have an accuracy of 89%. The neural net classifier performs 0.01 better with fake news in  both precision and f1 score but overall accuracy of the two is the same. Both appear to perform very well. 

Neural Net results 

label|precision|recall|f1-score  
|---|---|---|---|
FAKE |      0.90   |   0.88  |    0.89   |    
REAL   |    0.88   |   0.90   |   0.89   |   
accuracy   | -- |--|    0.89    |  

Logistic Regression results 

label|precision | recall |f1-score  
|---|---|---|---|
FAKE   |    0.89  |    0.88  |    0.88  |
REAL    |   0.88  |    0.90  |    0.89  |     
accuracy  |--| --|              0.89  |   


