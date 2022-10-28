# CS-433 Machine Learning - Project 1
The goal of this project is to train a machine learning model to detect a Higgs boson from a decay signature. For this purpose we build a binary classifier which takes as input a decay signature and labels it as coming from a Higgs boson or from another particle/process.
## Data
The data comes from the CERN particle accelerator and was made available on [aircrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files).
## Methods
- Linear regression using gradient descent
- Linear regression using stochastic gradient descent
- Least squares regression using normal equations
- Ridge regression using normal equations
- Logistic regression using gradient descent
- Regularized logistic regression using gradient descent
## File organization

* The implementations.py file contains all of our helper functions.
* The notebook file contains the data loading, preprocessing, model training and result generation for each of the above methods.
* The project1_description.pdf file contains project information.
* The run.py script runs our pipeline consisting of: data loading, preprocessing, model training and result generation for the best training model we have found experimentally.  
**Important note: the data is split at random in two sets for training and testing. This leads to some variability in the performance of each run.**
* The report.pdf file contains more precise information about how everything was created and tested and explains the choices we have made to try and improve the performance of our model.
  
 ## Running the model
In order to run run.py, it has to be at the same level as implementations.py and the test.csv and train.csv files found on aircrowd. This script will produce a file called output.csv with the predictions for test.csv.
If you also want to use the notebook file, the same file structure should be respected.
