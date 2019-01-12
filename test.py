# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 08:02:39 2019

@author: eviriyakovithya
"""


from datetime import datetime
import time
import pandas as pd
import pickle as pk
from bs4 import BeautifulSoup
import requests

link = condo_links_all[0]
page = requests.get(link)
print(link)

soup = BeautifulSoup(page.content, 'html.parser')
print(soup)

name=soup.find(class_="breadcrumb").findAll('span')[2].get_text()
print(name)

district=soup.find(class_="breadcrumb").findAll('span')[1].get_text()
print(district)

year_built=soup.find(class_="project-header-year").find('span').get_text()
print(year_built)

proj_area=soup.find(class_="project-header-area").find('span').get_text().split()[0]
print(proj_area)

nbr_buildings=soup.find(class_="project-header-tower").find('span').get_text()
print(nbr_buildings)

nbr_floors=soup.find(class_="project-header-floor").find('span').get_text()
print(nbr_floors)
