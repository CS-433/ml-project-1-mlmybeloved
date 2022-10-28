# M1 Machine Learning course project 1
This repo was made in the context of the CS433 ML course first project. The goal of this project is to train a machine learning model to tell whether the decay signature given by the input belongs to a Higgs boson or not.  
The data used is given on https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files and comes from the CERN particle accelerator.  
The implementations.py file contains all of our helper functions.  
The notebook file contains the data loading, preprocessing, model training and result generation.  
The project1_description.pdf file contains the actual project information.  
The run.py script runs the whole cycle of data loading, preprocessing, model training and result generation for the best training model we found in our notebook.  
The report.pdf file contains more precise information about how everything was created and tested and explains our choices in our model.  
  
In order to use the notebook file, you need to have implementations.py, the .csv files given on the aicrowd link (namely test.csv and train.csv) all in the same folder as the notebook itself.  
To run the run.py script, simply put it in the same folder as implementations.py and the .csv files (exactly like the notebook) and run it, it should write a .csv file called output.csv with the predictions for test.csv.
