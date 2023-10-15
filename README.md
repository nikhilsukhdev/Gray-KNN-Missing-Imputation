# Gray-KNN-Missing-Imputation
The main.py is used to perform an iterative missing value imputation using Gray KNN. The code is written in Python 3.9 using the below libraries/packages.

Description:-
	
The Gray KNN algorithm is based on the theory of Gray Relational Analysis, where gray distances are used to compare the similarity/dissimilarity between instances. Unlike the conventional KNN based imputation, Gray KNN can work with heterogeneous(numeric/categorical) attributes.

Getting Started
Dependencies:-

The following packages were used in Google colab.
1. Pandas 1.4.3
2. Numpy 1.23.1
3. Sklearn 1.1.1
4. OS (Built in module)
5. time (Built in module)
6. getopt (Built in module)
7. sys (Built in module)
8. warnings (Built in module)

Installing the packages:-
As the code would be running on a cloud platform (Google Colab) with all the dependencies in place, we do not need to install any of the above packages.

Executing the code:-

1. Please follow the steps in the report (Implementation Section) to open Google Colab. Once colab is opened, please copy the code from main.py file and paste it in the colab notebook.
2. Create the below 3 folders and upload the incomplete and complete datasets to the newly created folders as shown in the final report. 
	-Incomplete: Upload the incomplete .csv files to this folder
	-Complete: Upload the complete .csv files to this folder
	-Imputations: This folder would have your imputed .csv file
3. There are 4 parameters that need to be set as per the directions given in the report (Implementation Section). These are: -
	-base_path
	-incomplete_path
	-complete_path
	-k
4. Once the parameters are set, we are good to execute the code. Click on the run button, and you should be able to see the NRMS value in the output.

Please recch out on [Linkedin](https://linkedin.com/in/nikhil-sukhdev-882395183) for the datasets.
