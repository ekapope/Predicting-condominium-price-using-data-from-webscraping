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
from sklearn.preprocessing import RobustScaler

# import sklearn model class
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# import sklearn model selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# import sklearn model evaluation regression metrics
from sklearn.metrics import mean_squared_error

df=pd.read_csv(".\df_clean\df_cleaned.csv")
print(df.info())
print(df['district'].nunique())
#get dummies for district column
df = pd.get_dummies(df, columns=['district'])
#drop id, name, date columns
df = df.drop(['name', 'price_sqm'], axis=1)

print(df.info())

#group by id, check histrogram of count
counts_df = df.groupby('id').size().reset_index(name='counts')
print(counts_df.describe())
plt.hist(counts_df['counts'])
plt.show()
plt.hist(counts_df['counts'], cumulative=.1)
plt.show()
#from summary stats, looks ok to use time series method to predict
df= pd.merge(df,counts_df, how='left', on='id')

# change from long to wide
df_wide = df.pivot('date', 'id', 'value')


































# select all features to evaluate the feature importances
X = df.drop('price_sqm', axis=1)
X_colnames = X.columns
y = df[['price_sqm']]

# create scaler to the features
scaler = RobustScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# set up lasso regression to find the feature importances
lassoreg = Lasso(alpha=1e-5).fit(X_train, y_train)
feat = pd.DataFrame(data=lassoreg.coef_, \
                    index=X_colnames, \
                    columns=['FeatureImportances']).sort_values(['FeatureImportances'], \
                            ascending=False)


### linear regression model setup
md_lr = LinearRegression()

# linear regression model fit
md_lr.fit(X_train, y_train)

# linear regression model prediction
md_lr_y_pred = md_lr.predict(X_test)

# linear regression model metrics
md_lr_r_square = md_lr.score(X_test, y_test)

md_lr_mse = mean_squared_error(y_test, md_lr_y_pred) ** 0.5
md_lr_cvscores = np.sqrt(np.abs(cross_val_score(md_lr, X, y, cv=5,\
                                                       scoring='neg_mean_squared_error')))
print('linear regression\n  root mean squared error: %0.4f,\
      cross validation score: %0.4f (+/- %0.4f)' \
      %(md_lr_mse, md_lr_cvscores.mean(), 2 * md_lr_cvscores.std()))



### lasso regression model setup
md_lasso = Lasso(alpha=0.001)

# lasso regression md fit
md_lasso.fit(X_train, y_train)

# lasso regression md prediction
md_lasso_y_pred = md_lasso.predict(X_test)

# lasso regression md metrics
md_lasso_r_square = md_lasso.score(X_test, y_test)

md_lasso_mse = mean_squared_error(y_test, md_lasso_y_pred) ** 0.5
md_lasso_cvscores = np.sqrt(np.abs(cross_val_score(md_lasso, X, y, cv=5, scoring='neg_mean_squared_error')))
print('lasso regression\n  root mean squared error: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(md_lasso_mse, md_lasso_cvscores.mean(), 2 * md_lasso_cvscores.std()))


# specify the hyperparameter space
params = {'alpha': np.logspace(-4, 4, base=10, num=9)}

# lasso regression grid search model setup
model_lassoreg_cv = GridSearchCV(md_lasso, params, cv=5)

# lasso regression grid search model fit
model_lassoreg_cv.fit(X_train, y_train)

# lasso regression grid search model prediction
model_lassoreg_cv_ypredict = model_lassoreg_cv.predict(X_test)

# lasso regression grid search model metrics
model_lassoreg_cv_mse = mean_squared_error(y_test, model_lassoreg_cv_ypredict) ** 0.5
model_lassoreg_cv_cvscores = np.sqrt(np.abs(cross_val_score(model_lassoreg_cv, X, y, cv=5, scoring='neg_mean_squared_error')))
print('lasso regression grid search\n  root mean squared error: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_lassoreg_cv_mse, model_lassoreg_cv_cvscores.mean(), 2 * model_lassoreg_cv_cvscores.std()))
print('  best parameters: %s' %model_lassoreg_cv.best_params_)
