# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 07:18:59 2019

@author: eviriyakovithya
"""

# Import packages


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from datetime import datetime
import time
import pandas as pd
import pickle as pk

# Create an 'instance' of the driver.
# A new Chrome (or other browser) window should open up if options.headless = False (default)
CHROMEDRIVER_PATH = "./_chromedriver/chromedriver.exe"
options = Options()
options.headless = True
driver = webdriver.Chrome(CHROMEDRIVER_PATH, chrome_options=options)

# Enter the main page, all condos in Bangkok are grouped by district.
url ='https://www.hipflat.co.th/en/market/condo-bangkok-skik'
driver.get(url)

# Write function to scrape all links from the webpage.
def get_all_links(driver):
    links = []
    elements = driver.find_elements_by_class_name('directories__lists-element-name')
    for elem in elements:
        href = elem.get_attribute("href")
        links.append(href)  
    return links

# Run and store the links in district_links
start_time = datetime.now() 
district_links=get_all_links(driver)
time_elapsed = datetime.now() - start_time 
print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

# Check the length of 'district_links', there are 50 districts in Bangkok. 
# https://en.wikipedia.org/wiki/List_of_districts_of_Bangkok
print(len(district_links))

# Re-run the function to retrive all condo links in each district.
# Append to 'condo_links'.
start_time = datetime.now()
condo_links=[]
for district in district_links:
    print(len(condo_links),district)
#implicitly_wait - Specifies the amount of time the driver should wait 
#when searching for an element if it is not immediately present.
    driver.implicitly_wait(10)
    driver.get(district)
    condo_links.append(get_all_links(driver))
    time_elapsed = datetime.now() - start_time 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
print("completed")

# Now we got lists within a list (nested list)
# Turn a (nested) python list into a single list, that contains all the elements of sub lists
# Named as 'condo_links_all'
from itertools import chain
condo_links_all=list(chain.from_iterable(condo_links))
print("Total condo listings = "+str(len(condo_links_all)))
# Result in 2566 condo listings

# Dump the retrived links to text file.
with open("condo_links_all.txt", "w") as f:
    for s in condo_links_all:
        f.write(str(s) +"\n")
print("completed")


