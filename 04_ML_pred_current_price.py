# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 19:47:53 2019

@author: Chris
"""

import os
os.chdir(r"D:\GitHub_Personal\2019-01-Web-Scraping-using-selenium-and-bs4")

# import data manipulation library
import numpy as np
import pandas as pd

# import data visualization library
import matplotlib.pyplot as plt
import seaborn as sns

# import scientific computing library
import scipy

# import sklearn data preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.pipeline import Pipeline

# import sklearn model class
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor

# import sklearn model selection
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, cross_val_predict, validation_curve
# import sklearn model evaluation regression metrics
from sklearn import metrics
from sklearn.metrics import make_scorer,mean_squared_error, r2_score

###############################################################################
df=pd.read_csv(r"data\regression_data\df_cleaned_for_ML_regression.csv")
#goal is to predict current price, drop all duplicates
print(df.info())
print(df['district'].nunique())
print(df['tran_type1'].nunique())

# exclude everything with a price above or below 3 standard deviations (i.e. outliers)
df = df[np.abs(df["price_sqm"]-df["price_sqm"].mean())<=(3*df["price_sqm"].std())]

#get dummies for district column
#df = pd.get_dummies(df, columns=['district'])
df = pd.get_dummies(df, columns=['district'])
#drop id, name columns
df = df.drop(['id', 'name','bld_age',
              'tran_type1','tran_type2','tran_type3', 'tran_type4', 'tran_type5',
              'tran_name1','tran_name2', 'tran_name3', 'tran_name4', 'tran_name5'], axis=1)
#df = df.drop(['id', 'name'], axis=1)

print(df.info())
df['price_sqm'].describe()
plt.hist(df['price_sqm'])

corr_matrix = df.corr()
np.abs(corr_matrix["price_sqm"]).sort_values(ascending=False)

# select all features to evaluate the feature importances
X = df.drop('price_sqm', axis=1)
y = df['price_sqm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=121)

###############################################################################
# define function for output and plotting
def rmse_cv(model,n_folds=5):
    kf = KFold(n_folds, shuffle=True, random_state=121)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
#    cv_scores = cross_val_score(model, X, y, cv=kf)
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    r_sq_score = r2_score(y_test, y_pred)
    plt.scatter(y_test, y_pred)    
    plt.title(model)
    plt.xlabel("True prices")
    plt.ylabel("Predicted prices")  
#    plt.text(-1,220000, ' R-squared = {}'.format(float(cv_scores.mean())))
#    plt.text(-1,200000, ' R-squared Std = {}'.format(float(cv_scores.std())))
#    plt.text(-1,180000, ' MSE = {}'.format(round(float(mean_squared_error(y_test, predicted)), 2)))
    plt.show()
    return(rmse, r_sq_score)
###############################################################################
    
ols = make_pipeline(RobustScaler(), LinearRegression())
rmse,r_sq_score = rmse_cv(ols)
print('RMS: {:.2f}'.format(rmse.mean()),'r2_score: '+ str(r_sq_score))

###############################################################################

# manually tune alpha(s)
alpha_list =[0.0001,0.001,0.01,0.1,1,10,100]
result=[]
for alpha_val in alpha_list:
    ridge = make_pipeline(RobustScaler(), Ridge(alpha= alpha_val))
    rmse,r_sq_score = rmse_cv(ridge)
    print('RMS: {:.2f}'.format(rmse.mean()),'r2_score: '+ str(r_sq_score))
    result.append([alpha_val,np.mean(rmse),r_sq_score])
ridge_result = pd.DataFrame(result, columns = ['alpha_val','np.mean(rmse)','r_sq_score'])

###############################################################################

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =121)
rmse,r_sq_score = rmse_cv(GBoost)
print('RMS: {:.2f}'.format(rmse.mean()),'r2_score: '+ str(r_sq_score))