# Importing important libraries, such as pandas, numpy, seaborn, matplotlib
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Loading training dataset

df = pd.read_excel('trainset.xlsx') # Loading training dataset using pandas
df = df[df.CUST_AGE >= 60] # Filtering training dataset to customers who are 60 and above
df.head() # See preview of dataframe

# Exploratory Data Analysis

# Add new column called 'fraud_code' to convert FRAUD_NONFRAUD column into binary values (0: Fraud, 1: Nonfraud)
df.FRAUD_NONFRAUD = pd.Categorical(df.FRAUD_NONFRAUD)
df['fraud_code'] = df.FRAUD_NONFRAUD.cat.codes
# Convert columns with 'int64' data type to 'float64'
df.CUST_AGE = df.CUST_AGE.astype('float64')
df.OPEN_ACCT_CT = df.OPEN_ACCT_CT.astype('float64')
df.WF_dvc_age = df.WF_dvc_age.astype('float64')
# Convert several categorical columns into column with a numerical code for each category
# Converting these columns into numerical codes also handles missing values, as if there is a missing value, it will be given a code of -1
df.ALERT_TRGR_CD = pd.Categorical(df.ALERT_TRGR_CD)
df['alert_code'] = df.ALERT_TRGR_CD.cat.codes
df.AUTHC_PRIM_TYPE_CD = pd.Categorical(df.AUTHC_PRIM_TYPE_CD)
df['auth_code'] = df.AUTHC_PRIM_TYPE_CD.cat.codes
df.RGN_NAME = pd.Categorical(df.RGN_NAME)
df['rgn_code'] = df.RGN_NAME.cat.codes
df.AUTHC_SCNDRY_STAT_TXT = pd.Categorical(df.AUTHC_SCNDRY_STAT_TXT)
df['allow_code'] = df.AUTHC_SCNDRY_STAT_TXT.cat.codes
df.DVC_TYPE_TXT = pd.Categorical(df.DVC_TYPE_TXT)
df['device_code'] = df.DVC_TYPE_TXT.cat.codes
# Convert transaction timestamp column into datetime type to access the hour, month attributes
df['TRAN_TS'] = pd.to_datetime(df['TRAN_TS'])
# Add new columns for transaction hour and transaction month
df['tran_hour'] = df.TRAN_TS.dt.hour
df['tran_month'] = df.TRAN_TS.dt.month

# Data Visualizations
# Plotting various categorical and numerical columns against the fraud_code column to see if there's any correlation; uses seaborn library
sns.catplot(x=df.fraud_code,y=df.TRAN_AMT,data=df)
sns.catplot(x=df.fraud_code,y=df.ACCT_PRE_TRAN_AVAIL_BAL,data=df)
sns.catplot(x=df.fraud_code,y=df.OPEN_ACCT_CT,data=df)
sns.catplot(x=df.fraud_code,y=df.WF_dvc_age,data=df)
sns.catplot(y=df.RGN_NAME,hue="fraud_code",kind = "count",palette="pastel", edgecolor=".6",data=df)
sns.catplot(y=df.ALERT_TRGR_CD,hue="fraud_code",kind = "count",palette="pastel", edgecolor=".6",data=df)
sns.catplot(y="AUTHC_PRIM_TYPE_CD",kind = "count",data=df[df.fraud_code == 0])
sns.catplot(y=df.CUST_STATE,hue="fraud_code",kind = "count",palette="pastel", edgecolor=".6",data=df,height=8.27,aspect=11.7/8.27)
sns.catplot(y=df.AUTHC_SCNDRY_STAT_TXT,hue="fraud_code",kind = "count",palette="pastel", edgecolor=".6",data=df)
sns.catplot(y=df.DVC_TYPE_TXT,hue="fraud_code",kind = "count",palette="pastel", edgecolor=".6",data=df)
sns.catplot(y=df.TRAN_TS.dt.hour,hue="fraud_code",kind = "count",palette="pastel", edgecolor=".6",data=df)
sns.catplot(y=df.TRAN_TS.dt.dayofweek,hue="fraud_code",kind = "count",palette="pastel", edgecolor=".6",data=df)
sns.catplot(y=df.TRAN_TS.dt.month,hue="fraud_code",kind = "count",palette="pastel", edgecolor=".6",data=df)

# Building the Model

# Define a list of features from the dataframe
features = ['TRAN_AMT','ACCT_PRE_TRAN_AVAIL_BAL','WF_dvc_age','device_code']
# Create x and y arrays with features and fraud_code, respectively
x = np.array(df[features])
y = np.array(df['fraud_code'])
# Split the data into train and test sets with a 80/20 split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 50)

# Random Forests
from sklearn import metrics
reg_rf = RandomForestClassifier() # Initialize a random forest classifier
reg_rf.fit(X_train, y_train) # Fit the random forest classifier model onto the training set
y_pred = reg_rf.predict(X_test) # Predict values for fraud_code based on the features in X_test
print(metrics.classification_report(y_test, y_pred)) # Printing classification report
print("f1 score: ", metrics.f1_score(y_test, y_pred)) # Prints the F1 score
print("accuracy: ", metrics.accuracy_score(y_test,y_pred)) # Prints the accuracy
pd.crosstab(y_test, y_pred, rownames=['Actual Result'], colnames=['Predicted Result']) # Prints a cross table of predicted results vs actual results

feature_df = pd.DataFrame({'Importance':reg_rf.feature_importances_, 'Features': features }) # Creates a dataframe with feature importances in the model
print(feature_df)

# Running Model on Testset

dftest = pd.read_excel('testset.xlsx') # Read testset into a dataframe using pandas
dftest['FRAUD_NONFRAUD'] = "" # Create an empty column called FRAUD_NONFRAUD to be filled in later
# Create new columns for certain categorical variables to represent them numerically
dftest.ALERT_TRGR_CD = pd.Categorical(dftest.ALERT_TRGR_CD)
dftest['alert_code'] = dftest.ALERT_TRGR_CD.cat.codes
dftest.AUTHC_PRIM_TYPE_CD = pd.Categorical(dftest.AUTHC_PRIM_TYPE_CD)
dftest['auth_code'] = dftest.AUTHC_PRIM_TYPE_CD.cat.codes
dftest.DVC_TYPE_TXT = pd.Categorical(dftest.DVC_TYPE_TXT)
dftest['device_code'] = dftest.DVC_TYPE_TXT.cat.codes
# Convert a timestamp column into a datetime object to access the different attributes of it
dftest['TRAN_TS'] = pd.to_datetime(dftest['TRAN_TS'])
dftest['tran_hour'] = dftest.TRAN_TS.dt.hour
# Define features to be used in testset
dftestfeatures = ['ACCT_PRE_TRAN_AVAIL_BAL','WF_dvc_age','device_code','TRAN_AMT']
# Run model on defined features and store predicted fraud codes in an array
pred = reg_rf.predict(dftest[dftestfeatures])
# Fill values of FRAUD_NONFRAUD column with values from the array with predicted fraud codes
dftest['FRAUD_NONFRAUD'] = pred[:]
# Export final dataframe to excel sheet
dftest.to_excel('resulttable.xlsx')
dftestfinal = dftest[['dataset_id','FRAUD_NONFRAUD']].copy()
dftestfinal.to_excel('resulttablefiltered.xlsx')

# Sources
# https://towardsdatascience.com/building-classification-models-with-sklearn-6a8fd107f0c1
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# https://blogs.oracle.com/ai-and-datascience/post/an-introduction-to-building-a-classification-model-using-random-forests-in-python
# https://builtin.com/data-science/random-forest-algorithm