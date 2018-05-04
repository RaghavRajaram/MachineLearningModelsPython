import time
start_time = time.time()

import configparser
import warnings
import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn import neighbors
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

# Run the categorical encoder .py file. Should be placed in the main directory where data folder is placed.
%run CategoricalEncoder.py

# Transformer function 
class DataFrameSelector(BaseEstimator, TransformerMixin): 
    def __init__(self, attibute_names):
        self.attibute_names = attibute_names
    def fit(self, X, y=None):
        return(self)
    def transform(self, X): 
        return(X[self.attibute_names].values)

# Config reader    
config = configparser.ConfigParser()

# Ignore warnings
warnings.filterwarnings('ignore')

# Create an empty dataList to store the list of data given in 'datasets_selection'
dataList = []
selectedFiles = !cat datasets_selection

# Append each file name listed in datasets_selection
for i in selectedFiles:
    dataList.append(i[:-5])

# Function to separate the string list of categorical and value columns into list of integers. 
def strSeparator(strList):
    
    #Create an empty return list to return
    returnList = []
    for i in strList:
        
        #Create an int split list to append integer values of the split list members
        intSplitList = []
        
        if i == '':
            intSplitList.append(None)
        else:         
            #Split the input string list on ','
            splitList = i.split(',')
            
            #Add the integer value of each split member
            for a in splitList:
                intSplitList.append(int(a)-1)
        
        #Add the list of integers into the return list        
        returnList.append(intSplitList)
        
    return(returnList)


allFiles = !ls data 
type(allFiles)

# Find all separators, header rows, target, value and category columns in 54 files and store them in lists
separatorList = []
actualseparatorList = []
headerList = []
targetList = []
valueList = []
categoryList = []


for i in dataList:
    
    # Read the config file
    config.read('data/' + i+'/config.ini')
    
    if i == 'breast-cancer-wisconsin':
        config.read('data/breast-cancer-wisconsin-original/config.ini')
    
    # Handle separator values
    actual = config['info']['separator']
    if 'comma' in config['info']['separator']:
        fileSep = ','
    elif  config['info']['separator'] == '' :
        fileSep = 'space'
    else:
        fileSep = config['info']['separator']
    
    #Append actual and handled separators into respective lists
    actualseparatorList.append(actual)   
    separatorList.append(fileSep)
    
    #Append header from the config file into header list. Int handled
    headerList.append(int(config['info']['header_lines']))
    
    #Append target from the config file into target list. Int handled
    targetList.append(int(config['info']['target_index']))
    
    #Append value indices from the config file into value list. 
    valueList.append(config['info']['value_indices'])
    
    #Append category indices from the config file into category list. 
    categoryList.append(config['info']['categoric_indices'])
    
# convert the string list of value and category columns to int list    
valueList = strSeparator(valueList)
categoryList = strSeparator(categoryList)

# Create a dataframe to store the final accuracy values by datafile and algorithm names
finalDf = pd.DataFrame(columns=['DataFile','knn','logistic','svm','decision','quadratic','randomForest','AdaBoost'])

# current index variable to keep track of the loop 
count = 0

# Looping through each data file
for i in dataList:
    
    # Handle breast-cancer-wisconsin data which is stored in the 'breast-cancer-wisconsin-original' folder. 
    # Change Directory to the folder that contains the data file.
    
    if i == 'breast-cancer-wisconsin':
        os.chdir('data/breast-cancer-wisconsin-original/')
    else:
        mainpath = 'data/' + i + '/'
        os.chdir(mainpath)
    
    # List of files in the current directory.
    allFilesinFolder = !ls
    
    # If orig.custom file is available, use orig.custom else use orig file. 
    if i + '.data.orig.custom' in allFilesinFolder:
        fileName = i + '.data.orig.custom'
            
    else:
        fileName = i + '.data.orig'
     
    
    #read the data file with the corresponding separator and header. Handle empty value for separator through delim_whitespace
    #use na_values = ["?"] to handle '?' values and then finally drop all na. 
    
    if(separatorList[count] == 'space'):
        
        dataFile = pd.read_csv(fileName,delim_whitespace=True,skiprows=headerList[count],na_values=["?"],header=None,error_bad_lines=False)
    
    else:
        dataFile = pd.read_csv(fileName,sep=separatorList[count],skiprows=headerList[count],na_values=["?"],header=None,error_bad_lines=False)
        
    dataFile = dataFile.dropna()
    
    #Get Category, value and target column list of the datafile in use from the main list through the current index count. 
    
    categ = categoryList[count]
    val = valueList[count]
    targ = targetList[count]-1
    
    # if value columns not in category list, add them to newVal list which is the actual value column list to be used. 
    newVal = []
    for i in val:
        if i not in categ:
            newVal.append(i)
    
    # Define lists to contain category (cat), value (num) and target (out) columns from the dataFile. 
    cat_attributes = []
    num_attributes = []
    out_attribute = []
    
    #Loop through columns in the dataFile and rename them based on whether they are category, value or target columns. 
    #Drop columns that are not category, value or target columns. 
    
    for i in dataFile.columns:  
        
        if i in categ:
            dataFile = dataFile.rename(columns={i:'cat_' + str(i)})
            cat_attributes.append('cat_'+str(i))
        elif i in newVal :
            dataFile = dataFile.rename(columns={i:'num_' + str(i)})
            num_attributes.append('num_'+str(i))
        elif int(i) == int(targ):
            dataFile = dataFile.rename(columns={i:'out_' + str(i)})
            out_attribute.append('out_' + str(i))
        else:
            dataFile = dataFile.drop(i,axis = 1)
    
   
        
    # Pipeline function definitions   
    num_pipeline = Pipeline([
    ('selector',DataFrameSelector(num_attributes)),
    ('imputer',Imputer(strategy="median")),
    ('std_scaler', StandardScaler()),])

    cat_pipeline = Pipeline([
    ('selector',DataFrameSelector(cat_attributes)),
    ('label_binarizer',CategoricalEncoder(encoding="onehot-dense")),])

    # Pipeline assignment based on number of category and value attributes in the datafile. 
    if len(cat_attributes) == 0:
        full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline',num_pipeline),])
    elif len(num_attributes) == 0:
        full_pipeline = FeatureUnion(transformer_list=[
        ('cat_pipeline',cat_pipeline),])
    else:
        full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline',num_pipeline),
        ('cat_pipeline',cat_pipeline),])
       
    
    # Pass the dataFile through the transform function
    df = full_pipeline.fit_transform(dataFile)
    
    # Convert the transformed matrix into a data frame.
    df_ready = pd.DataFrame(df)
     
    # Reset the index of the target column 
    targetSeries = dataFile[out_attribute]
    targetSeries = targetSeries.reset_index()
    targetSeries = targetSeries.drop('index',axis = 1)
    
    # Add the target column to the transformed data frame.
    df_ready['target'] = targetSeries
   
    train_set, test_set = train_test_split(df_ready, test_size=0.2, random_state=1) #random_state is used to set the SEED
    file = train_set.copy()     #file now represents our train set which is 80% of the original file data 
        
    # 2. Initialize the input and output variables
    y = file['target']  
    X = file.drop('target',axis=1)     
   
    test_y = test_set['target']
    test_x = test_set.drop('target',axis=1)  

    # fit a KNN classifier to the entire dataset
    clf = neighbors.KNeighborsClassifier(1)               
    clf.fit(X, y) 
    # built-in function for computing accuracy
    knnScore = (clf.score(test_x, test_y))

    #Logistic Regression
    clf = linear_model.LogisticRegression() 
    clf.fit(X, y)     
    lrScore = clf.score(test_x, test_y)

    #Support Vector Machine
    clf = svm.SVC()
    clf.fit(X, y)  
    svmScore = clf.score(test_x, test_y)

    #Decision Tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)  
    decisionScore = clf.score(test_x, test_y)

    #Quadratic Discriminant Analysis 
    clf = QuadraticDiscriminantAnalysis()                                 
    clf.fit(X, y)  
    quadScore = (clf.score(test_x, test_y))

    #Random Forests
    clf = RandomForestClassifier()
    clf.fit(X, y)
    rfScore = clf.score(test_x, test_y)

    #AdaBoost
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(X,y)
    adaScore = clf.score(test_x, test_y)
    
    # Insert a new row containing the dataFile name and accuracy values into the Final table. 
    finalDf.loc[count] =  [dataList[count],knnScore,lrScore,svmScore,decisionScore,quadScore,rfScore,adaScore]


    # Change directory back to the main directory. (Two directories up)
    os.chdir("../..")
    
    # Add 1 to current index to iterate. 
    count = count + 1


print(finalDf)

# Write final table with datafile names and accuracy measures for each classification algorithm to a csv named 'FinalDF'.
finalDf.to_csv("FinalDF.csv",index = False)
elapsed_time = time.time() - start_time

print('Time Elapsed: '+ str(elapsed_time) + 'seconds') #Took 20s on an average