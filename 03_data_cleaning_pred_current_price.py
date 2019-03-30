# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:41:39 2019

@author: Chris
"""

# import packages
import re
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

#add building age 2019-'year_built'
df['bld_age'] = 2019-df['year_built']

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

def find_dist(input_str):
    input_str = str(input_str)
    dist = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", input_str)[0]
    unit = re.findall(r" km ", input_str)
    if (len(unit)!=0): dist_km = float(dist)
    else: dist_km = float(dist)/1000
    return(dist_km)
    
#check shop col
df['shops']=df['shops'].str.replace('\'','').str.split('\,')
df['shops'][0]
col_list = ['dist_shop_'+str(i) for i in range(1, 6)]
# expand list into its own dataframe
df[col_list]= df['shops'].apply(pd.Series)
# loop all columns, result in distance (km)
for col in col_list: df[col]=df[col].apply(lambda x: find_dist(x))

#check schools col
df['schools']=df['schools'].str.replace('\"','\'').str.split('\', \'')
df['schools'][0]
len_school = df['schools'].apply(lambda x: len(x))
# expand list into its own dataframe
col_list = ['dist_school_'+str(i) for i in range(1, 6)]
df[col_list]= df['schools'].apply(pd.Series)
# loop all columns, result in distance (km)
for col in col_list: df[col]=df[col].apply(lambda x: find_dist(x))

#check restaurants col
df['restaurants']=df['restaurants'].str.replace('\"','\'').str.split('\', \'')
df['restaurants'][0]
col_list = ['dist_food_'+str(i) for i in range(1, 6)]
# expand list into its own dataframe
df[col_list]= df['restaurants'].apply(pd.Series)
# loop all columns, result in distance (km)
for col in col_list: df[col]=df[col].apply(lambda x: find_dist(x))

#check hospital col
df['hospital'][0]
df['hospital']=df['hospital'].apply(lambda x: find_dist(x))

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

#price_sqm
plt.hist(df['price_sqm'])
plt.show()
print(df['price_sqm'].describe())

# check transportation col
df['transportation']=df['transportation'].str.replace('\'','').str.split('\,')
# check the split
len_chk = df['transportation'].apply(lambda x: len(x))
df['transportation'][0]
# element number 1,4,7,10,13 are station names
col_list = ['tran_name'+str(i) for i in range(1, 6)]
df[col_list]= df['transportation'].apply(pd.Series).iloc[:,[1,4,7,10,13]].apply(lambda x: x.str.strip())
# element number 2,5,8,11,14 are distance to station
col_list = ['dist_tran_'+str(i) for i in range(1, 6)]
# expand list into its own dataframe
df[col_list]= df['transportation'].apply(pd.Series).iloc[:,[2,5,8,11,14]]
# loop all columns, result in distance (km)
for col in col_list: df[col]=df[col].apply(lambda x: find_dist(x))

#drop id, name, date columns
df.drop(['year_built','shops', 'schools', 'restaurants', 'amenities',
         'transportation','change_last_q', 'change_last_y', 'rental_yield',
         'change_last_y_rental_price', 'price_hist'], axis=1, inplace=True)

df.isnull().sum()
df.info()

#export to csv
df.to_csv("df_cleaned_for_ML_regression.csv", index=False, encoding='utf-8-sig')
