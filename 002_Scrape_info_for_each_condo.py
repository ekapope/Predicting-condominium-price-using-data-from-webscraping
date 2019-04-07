# -*- coding: utf-8 -*-
"""
*** package bs4 is needed for this script.

This script will:
1. load all links from 'condo_links_all.txt', store as a list
2. extract attributes for each link using retrieve function
3. retrieve function will check if there is any historical data available or not
    if there is no historical data, it will skip to next link (line # 41)
4. 5 seconds sleep time was set between each request (line # 126)
5. save output as 'df_completed.csv'
"""

# import packages
from datetime import datetime
import time
import pandas as pd
import pickle as pk
from bs4 import BeautifulSoup
import requests
import os
os.chdir(r"D:\GitHub_Personal\2019-01-Web-Scraping-using-selenium-and-bs4")

# open the output text file
with open('condo_links_all.txt') as f:
    condo_links_all = f.read().splitlines()
print(len(condo_links_all))

##############################################################################
# Write function to retrive info, using bs4.
# This process took some time to carefully extract the info you needed from the soup.

def retrieve(link):
    page = requests.get(link)
    print(link)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    graph_data=soup.find(id="graph1").get_text().strip()
    
    # If there is no price chart in the page, return 'None', skip the listing
    if graph_data != "Not enough data to build the graph":
        name=soup.find(class_="breadcrumb").findAll('span')[2].get_text()
        district=soup.find(class_="breadcrumb").findAll('span')[1].get_text()
        latitude=str(soup.find(itemprop="latitude")).split("=")[1].split()[0]
        longitude=str(soup.find(itemprop="longitude")).split("=")[1].split()[0]
    
        description=str(soup.find(class_="property-description__content"))
        year_built=soup.find(class_="project-header-year").find('span').get_text()
        proj_area=soup.find(class_="project-header-area").find('span').get_text().split()[0]
        nbr_buildings=soup.find(class_="project-header-tower").find('span').get_text()
        nbr_floors=soup.find(class_="project-header-floor").find('span').get_text()
        units=description.split("units")[0].split()[-1]
        print(name,district,latitude,longitude,"\n",year_built,proj_area,nbr_buildings,nbr_floors,units)
        
        neighborhood=[]
        for i in range(0,15):
            x=soup.find(class_="property-description__content").findAll('li')[i].get_text()
            neighborhood.append(x)
        
        shops=neighborhood[0:5]
        #for x in shops: print(x)
            
        schools=neighborhood[5:10]
        #for x in schools: print(x)
        
        restaurants=neighborhood[10:15]
        #for x in restaurants: print(x)
        
        hospital=soup.find(class_="property-description__content").findAll('p')[-3].get_text()
        #print(hospital)
        
        # Amenities section
        # Elevator,Parking,Security,CCTV,Pool,Sauna,Gym,Garden,Playground,Shop,Restaurant,Wifi
        amenities=[]
        for i in range(0,12):
            if ('yes' in str(soup.find(class_="amenities").findAll('li')[i])):
                amenities.append(1)
            else:
                amenities.append(0)
        #print(amenities)
        
        # Location and Neighborhood
        transportation=[]
        for i in range(0,5):
            tran_type=soup.findAll(class_="media neighborhood-destination")[i].find(class_="icon").i['class'][1]
            trans_name=soup.findAll(class_="media-heading")[i].get_text()
            trans_dist=soup.findAll(class_="media neighborhood-destination")[i].find('small').get_text()
            transportation.append((tran_type,trans_name,trans_dist))
            
        # Market Stats
        price_sqm=soup.find(class_="indicator__amount").find(class_="money").get_text().strip('฿').replace(',',"")
        change_last_q=soup.findAll(class_="indicator__amount")[1].get_text().replace('\n',"").strip()
        change_last_y=soup.findAll(class_="indicator__amount")[2].get_text().replace('\n',"").strip()
        rental_yield=soup.findAll(class_="indicator__amount")[3].get_text().replace('\n',"").strip()
        change_last_y_rental_price=soup.findAll(class_="indicator__amount")[4].get_text().replace('\n',"").strip()
        #print(price_sqm,change_last_q,change_last_y,rental_yield,change_last_y_rental_price)
        
        # price history graph
        price_hist=soup.find(class_="row-fluid background-color-gray project__graph-container").find('script').get_text().split('\n')[3].strip().strip(',').replace('data: ',"")
        #print(price_hist)
        
        return (name,district,latitude,longitude,year_built,proj_area,nbr_buildings,nbr_floors,units,\
           shops,schools,restaurants,hospital,amenities,transportation,\
           price_sqm,change_last_q,change_last_y,rental_yield,change_last_y_rental_price,price_hist)
    else:
        print("---------Not enough data to build the graph----------",'\n')
        
##############################################################################
# Run the loop to retrieve data and store data as DataFrame, save as pickle.

start_time = datetime.now()

condo_list=[]
i=0

for link in condo_links_all:
    try:
        condo_list.append(retrieve(link))
    except Exception: # Let the codes go if there is any error.
        pass
    print(i)
    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    
    ### Give the 'sleep' time = 5 seconds. Space out each request so the server isn’t overwhelmed.
    time.sleep(5)
    i=i+1

    # This is the preventive step...
    # You can even clear the list and name a new file to save processing memory.
    # Dump the data periodically every 5 iterations.
    if (i%5==0):
        # Delete 'None' elements from the list.
        condo_list = [c for c in condo_list if c is not None]
        df = pd.DataFrame(condo_list)
        with open('df.pkl', 'wb') as f:
            pk.dump(df, f)
        # Print out i,len(condo_list), so we can trace back if error occur.
        # i is the index of 'condo_links_all'
        print('------------------------ dump @ i = ',i,len(condo_list))         
print("completed")

# Once complete, dump to pickle and save as 'df_completed.pkl'.
condo_list = [c for c in condo_list if c is not None]
df_completed = pd.DataFrame(condo_list)
with open('df_completed.pkl', 'wb') as f:
    pk.dump(df_completed, f)
    
# export to csv
col_names= ['name','district','latitude','longitude','year_built','proj_area','nbr_buildings','nbr_floors','units',
            'shops','schools','restaurants','hospital','amenities','transportation',
            'price_sqm','change_last_q','change_last_y','rental_yield','change_last_y_rental_price','price_hist']
df_completed.to_csv("df_completed.csv" ,header=col_names,index=False,encoding='utf-8-sig')

#load csv
df_dirty= pd.read_csv("df_completed.csv", sep=',',encoding='utf-8-sig')