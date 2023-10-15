import pandas as pd
import numpy as np
import os
import time
from sklearn.base import TransformerMixin
import getopt
import sys
import warnings

warnings.filterwarnings("ignore")  # Ignore warnings

# Remove 1st argument from the
# list of command line arguments
argumentList = sys.argv[1:]

base_path = ''
incomplete_path = ''
complete_path = ''
k = 0
# Options
options = "b:i:c:k:"

# Long options
long_options = ["base_path", "incomplete_ds", "complete_ds", "k_neighbours"]

try:
    # Parsing argument
    arguments, values = getopt.getopt(argumentList, options, long_options)

    # checking each argument
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-b", "--base_path"):
            base_path = currentValue
        elif currentArgument in ("-i", "--incomplete_ds"):
            incomplete_path = base_path + '/' + currentValue
        elif currentArgument in ("-c", "--complete_ds"):
            complete_path = base_path + '/' + currentValue
        elif currentArgument in ("-k", "--k_neighbours"):
            k = int(currentValue)

except getopt.error as err:
    # output error, and return with an error code
    print(str(err))


class meanModeImputer(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, data, y=None):
        self.fill = pd.Series([data[c].value_counts().index[0]
                               if data[c].dtype == np.dtype('O') else data[c].mean() for c in data],
                              index=data.columns)
        return self

    def transform(self, data, y=None):
        return data.fillna(self.fill)


# GKnn Algorithm

def GKnn(impute_df):

    for i in range(len(empty_loc_cols) - 1):
        ## Differentiate numerical and categorical data
        numerical_cols = impute_df._get_numeric_data().columns #fetches just the numeric columns
        cols = impute_df.columns                                 # entire columns in dataset
        categorical_cols = list(set(cols) - set(numerical_cols)) # gives us just the categorical columns
        num_df = impute_df.loc[:,numerical_cols]                 #locates all rows in numerical dataframe column by their labels
        cat_df = impute_df.loc[:,categorical_cols]               #locates all columns in categorical dataframe column by their labels

        num_ck=num_df.iloc[empty_loc_cols[i],:]                  # gets the location/index of emptyelements in a numeric DF

        maxIter = 1
        num_cp=num_df.copy()
        if i+1 in num_cp.index:
            num_cp.drop([i + 1], axis = 0)                           #dropping adjacent emptyrows on indexing for numerical dataset to avoid duplication

        cat_ck=cat_df.iloc[empty_loc_cols[i],:]                  # gets the location/index of empty elements in a categorical DF
        cat_cp=cat_df.copy()
        if i+1 in cat_cp.index:
            cat_cp.drop([i + 1], axis = 0)                           # dropping adjacent emptyrows on indexing for categorical dataset to avoid duplictaion

        def greyRelationalGrade(impute_df):

            t1=pd.DataFrame() #initializing a dataframe
            t2=pd.DataFrame(data = None, columns=cat_cp.columns, index=cat_cp.index) #has the shape of the cat_cp dataframe

            for j in range(num_cp.index.size):
                temp = pd.Series(num_cp.iloc[j,:]-num_ck) #subtract all records of a feature with mean of that feature
                t1 = t1.append(temp,ignore_index=True) #append DF's t1 & temp

                if(cat_cp.iloc[j,:].equals(cat_ck)):
                    t2.iloc[j,:] = 1
                else:
                    t2.iloc[j,:] = 0

            t1.reset_index(drop=True, inplace=True) #Df after subtracting all records with mean

            t2.reset_index(drop=True, inplace=True)
            t = pd.concat([t1,t2], axis=1)
            # t = pd.concat(t1, axis=1)

            mmax=t1.abs().max().max()
            mmin=t1.abs().min().min()
            rho=0.5

            # Computing grey correlation coefficient
            GRC=((mmin+rho*mmax)/(abs(t1)+rho*mmax))

            # Computing the GRG and getting the rank
            GRG=GRC.sum(axis=1)/GRC.columns.size
            GRGSort = GRG.sort_values(ascending= False)

            return GRGSort

        count = 0
        prev = pd.DataFrame(data=incomplete_df.iloc[empty_loc_cols]).mean() # mean of instance which was missing before meanmode imputation
        for j in range(len(empty_loc[0])-1):
            result= greyRelationalGrade(impute_df)
            resultSorted = result.sort_values(ascending = False)
            neighbours = resultSorted.iloc[0:k].index
            # find the mean , and most frequent of the values using the index of these instances
            values = impute_df.iloc[neighbours]
            num_neigh= values._get_numeric_data().columns
            cat_neigh = list(set(values) - set(num_neigh))
            cat_neigh_cols = values.loc[:, cat_neigh]
            num_neigh_cols = values.loc[:,num_neigh]
            nMean = num_neigh_cols.mean(axis = 0)
            if not cat_neigh_cols.empty:
                nMode = cat_neigh_cols.value_counts().index
            ## imputation using k nearest neighbor instances
            if empty_loc_cols[i] == empty_loc[0][j]:
                if empty_loc_cols[i] in num_neigh_cols.index.get_level_values(0):
                    impute_df.iloc[empty_loc[0][j], empty_loc[1][j]] = nMean[len(empty_loc_col)-1]
                elif empty_loc_cols[i] in cat_neigh_cols:
                    impute_df.iloc[empty_loc[0][j], empty_loc[1][j]] = nMode[len(empty_loc_col)-1]
            count =+ 1
            tol = 10e-4

            ##convergence criteria
            if count >= maxIter or max(abs(impute_df.iloc[empty_loc_cols].mean() - prev)) <= tol:
                break

            prev = impute_df.iloc[empty_loc_cols].mean()
    return impute_df


#Frobenius Function
def frob(x):
    return np.linalg.norm(x, 'fro')

#NRMS
def nrms(original, imputed):
    x = frob(imputed - original) / frob(original)
    return x

## import xlsx files as pandas dataframe
complete_df = pd.read_csv(complete_path)
incomplete_df_name_list = []
incomplete_df_list = []
for filename in os.listdir(incomplete_path):
    if filename.endswith('.csv'):
        #if filename.startswith('Data_1_AE_1%'):
        #or filename.startswith('Data_1_AG_5%'):
        #if filename.startswith('Data_6_NE_10%'):
        incomplete_df_list.append(pd.read_csv(incomplete_path+"/"+filename, header=None))
        incomplete_df_name_list.append(filename.split(".")[0])

#imputed_df_list = []
nrms_data = []
#ae_data = {}
index = 0
def calc_time(new_time):
    """
    :param new_time:
    :return:
    """
    min=new_time//60
    sec=new_time%60
    t=(min,sec)

    return t

for incomplete_df in incomplete_df_list:

    incomplete_df.reset_index()
    complete_df.reset_index()
    complete_df.columns=["Col"+str(i) for i in range(complete_df.shape[1])] # gives an arrary size of column count
    incomplete_df.columns=["Col"+str(i) for i in range(incomplete_df.shape[1])] # gives array size of column count(incompleteDF)

    ## find empty locations
    empty_loc = np.where(pd.isnull(incomplete_df)) #([4,4],[0,1])
    empty_loc_cols = np.unique(empty_loc[0]) #[4]
    empty_loc_col = np.unique(empty_loc[1]) #[0,1]

    ## Replace NaN values to mean values
    incomplete_df = meanModeImputer().fit_transform(incomplete_df)
    print('Imputing File...', incomplete_df_name_list[index])
    print('Starting Imputation...')
    start_time=time.time()
    ## GKnn
    imputed_df = GKnn(incomplete_df)
    print('Ended Imputation...')
    complete_time=time.time()
    total_time=complete_time-start_time
    final_time=calc_time(total_time)
    mins,secs=final_time
    print("Total time taken is: {} mins and {} secs".format(mins,secs))
    numerical_cols = imputed_df._get_numeric_data().columns
    cols = imputed_df.columns
    categorical_cols = list(set(cols) - set(numerical_cols))
    cat_empty_loc = np.where(pd.isnull(incomplete_df.loc[:,categorical_cols]))
    imputed_num_df = imputed_df.loc[:,numerical_cols]        #numerical dataframe of the imputed data
    imputed_cat_df = imputed_df.loc[:,categorical_cols]      #categorical dataframe of the imputed data
    complete_num_df = complete_df.loc[:,numerical_cols]        #numerical dataframe of the original data
    complete_cat_df = complete_df.loc[:,categorical_cols]      # categorical dataframe of the original data
    #Compute NRMS
    nrmser = nrms(complete_num_df, imputed_num_df)
    print('NRMS :: ',nrmser)
    nrms_data.append({incomplete_df_name_list[index]: nrmser})
    ## Compute AE
    imputed_ae = 0
    for x in range(len(cat_empty_loc[0]) - 1):
        if(imputed_cat_df.iloc[cat_empty_loc[0][x], cat_empty_loc[1][x]] == complete_cat_df.iloc[cat_empty_loc[0][x], cat_empty_loc[1][x]]):
            imputed_ae+=1
    AE = (complete_cat_df.size - len(cat_empty_loc[0]) + imputed_ae) / complete_cat_df.size
    #ae_data.update( {incomplete_df_name_list[index]: AE} )
    #imputed_df_list.append(imputed_df)
    imputed_df.to_csv(base_path + '/Imputations/' +incomplete_df_name_list[index]+"_imputed.csv") #save imputed dataframes to csv
    index+=1

print(nrms_data)
#print(ae_data)