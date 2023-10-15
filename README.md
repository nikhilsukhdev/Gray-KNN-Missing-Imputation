# Gray-KNN-Missing-Imputation
The main.py is used to perform an iterative missing value imputation using Gray KNN. The code is written in Python 3.9 using the below libraries/packages.

Description:-
	
The Gray KNN algorithm is based on the theory of Gray Relational Analysis, where gray distances are used to compare the similarity/dissimilarity between instances. Unlike the conventional KNN based imputation, Gray KNN can work with heterogeneous(numeric/categorical) attributes.

Getting Started
Dependencies:-

The following packages were used in Google colab.
•Pandas 1.4.3
•Numpy 1.23.1
•Sklearn 1.1.1
•OS (Built in module)
•time (Built in module)
•getopt (Built in module)
•sys (Built in module)
•warnings (Built in module)

Installing the packages:-
•As the code would be running on a cloud platform (Google Colab) with all the dependencies in place, we do not need to install any of the above packages.


Executing the code:-

• Please follow the steps in the report (Implementation Section) to open Google Colab. Once colab is opened, please copy the code from main.py file and paste it in the colab notebook.
• Create the below 3 folders and upload the incomplete and complete datasets to the newly created folders as shown in the final report. 
• Incomplete: Upload the incomplete .csv files to this folder
• Complete: Upload the complete .csv files to this folder
• Imputations: This folder would have your imputed .csv file
• There are 4 parameters that need to be set as per the directions given in the report (Implementation Section). These are: -
• base_path
• incomplete_path
• complete_path
• k
• Once the parameters are set, we are good to execute the code. Click on the run button, and you should be able to see the NRMS value in the output.
