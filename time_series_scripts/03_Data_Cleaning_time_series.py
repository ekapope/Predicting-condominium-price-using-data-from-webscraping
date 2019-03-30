# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:41:39 2019

@author: Chris
"""

# import packages

from datetime import datetime
import time
import pandas as pd
import pickle as pk
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import requests
import os
os.chdir(r"D:\GitHub_Personal\2019-01-Web-Scraping-using-selenium-and-bs4")

#load csv
df= pd.read_csv("df_completed.csv", sep=',',encoding='utf-8-sig')

df.isnull().sum()
df.info()

#add id column
df['id'] = df.index

#clean up each column
#strip str
df['name'] = df['name'].str.strip()
df['district'] = df['district'].str.strip()

#replace " and change to numeric
df['latitude'] = pd.to_numeric(df['latitude'].str.replace('\"',''))
df['longitude'] = pd.to_numeric(df['longitude'].str.replace('\"',''))

#replace , and change to numeric
df['proj_area'] = pd.to_numeric(df['proj_area'].str.replace('\,',''))

#check no numeric rows, convert to numeric, replace with Nan
print(df[~df['units'].str.isnumeric()]['units'])
df['units'] = pd.to_numeric(df['units'], errors = 'coerce')
missing_idx = np.isnan(df['units'])
missing_unit_df = df[missing_idx]
plt.hist(df['units'][~missing_idx])
plt.show()
print(df['units'][~missing_idx].describe())
#fill na with median in the same district
df['units'] = df.groupby("district")["units"].transform(lambda x: x.fillna(x.median()))

#check shop col
df['shops']=df['shops'].str.replace('\'','').str.split('\,')

#check schools col
df['schools']=df['schools'].str.replace('\'','').str.split('\,')

#check restaurants col
df['restaurants']=df['restaurants'].str.replace('\'','').str.split('\,')

#check hospital col
df['hospital']=df['hospital'].str.replace('\'','').str.split('\,')

#amenities col
#split into columns
#Elevator,Parking,Security,CCTV,Pool,Sauna,Gym,Garden,Playground,Shop,Restaurant,Wifi
col_list = ['Elevator','Parking','Security','CCTV','Pool','Sauna','Gym',\
            'Garden','Playground','Shop','Restaurant','Wifi']
df['amenities']=df['amenities'].str.replace('\'','')
len(df['amenities'][0])
df['amenities'][0]
df[col_list] = df['amenities'].str.split(",",expand=True,)
df[col_list] = df[col_list].apply(lambda x: x.str.strip())
df[col_list] = df[col_list].apply(lambda x: x.str.replace('\[','')).\
apply(lambda x: x.str.replace('\]',''))
df[col_list] = df[col_list].apply(pd.to_numeric)

#transportation 
df['transportation']=df['transportation'].str.replace('\'','').str.split('\,')
df['transportation'][0]

#price_sqm
plt.hist(df['price_sqm'])
plt.show()
print(df['price_sqm'].describe())

##change_last_q
#plt.hist(df['change_last_q'])
#plt.show()
#print(df['change_last_q'].describe())
#
##change_last_y
#df['change_last_y'].head()
#non_num_idx = pd.to_numeric(df['change_last_y'], errors='coerce').isnull()
#print(df['change_last_y'][non_num_idx])
#
##plt.hist(df['change_last_y'])
#plt.show()
#print(df['change_last_q'].describe())

#drop unwanted columns for now
#df = df[['id', 'name', 'district', 'latitude', 'longitude', \
#         'year_built', 'proj_area', 'nbr_buildings', 'nbr_floors', 'units', \
#         'price_sqm', 'price_hist',\
#         'Elevator', 'Parking', 'Security', 'CCTV', 'Pool', 'Sauna',\
#         'Gym', 'Garden', 'Playground', 'Shop', 'Restaurant', 'Wifi']]
#price_hist
from ast import literal_eval
df['price_hist'] = df['price_hist'].apply(lambda x : literal_eval(x))
#for each row, change dict to df and concat to new df
#expand and save into chunks        
nrows = len(df)
temp_df = pd.DataFrame(columns=df.columns)
start_time = datetime.now()
for i in range(nrows):
    date_value_df = pd.DataFrame(df['price_hist'][i])
    date_value_df['id'] = i
    this_id_df = pd.merge(df,date_value_df,how='right',on='id')
    temp_df = pd.concat([temp_df,this_id_df],sort=True)
    if(i%100==0 or i == nrows-1):
        temp_df.to_csv(".\df_clean\df_"+str(i)+".csv" ,\
                       header=temp_df.columns,\
                       index=False, encoding='utf-8-sig')
        temp_df = pd.DataFrame(columns=df.columns)
        time_elapsed = datetime.now() - start_time
        print(str(i)+' Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

#combine all chunks
import os
import glob
os.chdir(r"D:\GitHub_Personal\2019-01-Web-Scraping-using-selenium-and-bs4\df_clean")

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])

#drop unwanted columns for now
combined_csv = combined_csv[['id', 'name', 'district', 'latitude', 'longitude', \
         'year_built', 'proj_area', 'nbr_buildings', 'nbr_floors', 'units', \
         'price_sqm', 'date', 'value',\
         'Elevator', 'Parking', 'Security', 'CCTV', 'Pool', 'Sauna',\
         'Gym', 'Garden', 'Playground', 'Shop', 'Restaurant', 'Wifi']]

#export to csv
combined_csv.to_csv("df_cleaned.csv", index=False, encoding='utf-8-sig')



#df.info()
#df= pd.read_csv("df_completed.csv", sep=',',encoding='utf-8-sig')
##add id column
#df['id'] = df.index
#a = range(len(df))
















##impute missing # units, using linear regression with 4 columns
#from sklearn import linear_model
#from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
#from sklearn.model_selection import train_test_split
#pred_units = df[['year_built', 'proj_area', 'nbr_buildings', 'nbr_floors', 'units']][~np.isnan(df['units'])]
#
##Split the data into training/testing sets
##X = pred_units[['year_built', 'proj_area', 'nbr_buildings', 'nbr_floors']]
#X = pred_units[['proj_area', 'nbr_buildings', 'nbr_floors']]
#y= pred_units['units']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
#
## Create linear regression object
#regr = linear_model.LinearRegression()
#
## Train the model using the training sets
#regr.fit(X_train, y_train)
#
## Make predictions using the testing set
#y_pred = regr.predict(X_test)
#
#accuracy = regr.score(X_test,y_test)
#
## The coefficients
#print('Coefficients: \n', regr.coef_)
## The mean squared error
#print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
## Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(y_test, y_pred))

