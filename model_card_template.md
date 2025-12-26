# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is implemented using a Logistic Regression algorithm from scikit-learn.  


## Intended Use
The intended use of this model is educational and demonstrative.  
## Training Data
The training data is derived from the UCI Census Income (Adult) dataset, stored locally as `census.csv`.  
The dataset includes demographic and employment-related attributes such as age, workclass, education, marital status, occupation, race, sex, and native country.

## Evaluation Data
The evaluation data consists of the held-out 20% test split from the same census dataset.
## Metrics
The model was evaluated using the following classification metrics:

- **Precision**
- **Recall**
- **F1 Score**
On the test dataset, the model achieved the following performance:

- **Precision:** 0.7390  
- **Recall:** 0.5731  
- **F1 Score:** 0.6456  
## Ethical Considerations
This model is trained on demographic data and may reflect historical and societal biases present in the dataset.
## Caveats and Recommendations
The model uses a relatively simple Logistic Regression algorithm and does not include advanced feature scaling or hyperparameter optimization. 
Future improvements could include:
- More robust bias analysis and mitigation techniques
This model should be used strictly for learning and demonstration purposes.