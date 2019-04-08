# Predicting condominium price using data from webscraping


### 1.	Data set and explanation webscraping process
This project uses Selenium library to firstly obtain all condominiums listed on the https://www.hipflat.com/ website, and extracts information for each page using BeautifulSoup package. Hipflat is one of the biggest property listing website in Thailand. This project is focused on condominium listings in Bangkok, both new and resale. Refer to below links for Python scripts.

[001_Retrieve_all_urls.py](https://github.com/ekapope/2019-01-Web-Scraping-using-selenium-and-bs4/blob/master/001_Retrieve_all_urls.py)

[002_Scrape_info_for_each_condo.py](https://github.com/ekapope/2019-01-Web-Scraping-using-selenium-and-bs4/blob/master/002_Scrape_info_for_each_condo.py)


### 2.	Pre-processing and data cleaning
Check NAs and data types for each column. Perform data manipulation by clean each column using regex, change numbers from strings to numeric, impute missing values, and convert lists of strings into columns. Refer to the link below.

[003_data_cleaning_pred_current_price.py](https://github.com/ekapope/2019-01-Web-Scraping-using-selenium-and-bs4/blob/master/003_data_cleaning_pred_current_price.py)


### 3.	Data scaling and hyperparameter tuning & ML
Robust Scaler  is used in the pipeline before passing through the ML models. It uses a similar method to the Min-Max scaler but it instead uses the interquartile range, rather than the min-max, so that it is robust to outliers. 

[004_1_ML_pred_current_price.py](https://github.com/ekapope/2019-01-Web-Scraping-using-selenium-and-bs4/blob/master/004_1_ML_pred_current_price.py)


### Three machine learning algorithms were used in the project
1. Ridge
2. RandomForestRegressor
3. GradientBoostingRegressor

The results are shown below.
![Result table](https://github.com/ekapope/2019-01-Web-Scraping-using-selenium-and-bs4/blob/master/ipython_files/result_table.PNG "Result table")


![Scatter plot the result of Gradient Boosting Regressor](https://github.com/ekapope/2019-01-Web-Scraping-using-selenium-and-bs4/blob/master/ipython_files/qt_img121839632252933.png "Gradient Boosting Result")


# Summary and suggestion for future improvement

Even this dataset is quite small with lots of features and we can only predict the price per square meters for each condo, however, this study is very useful for buyers, resellers, agents and even developers to justify the 'fair price' as a starting point based on the current actual market data.


In the webscraping step, we should acquire all listings available in each condo, not only average price per sqm. This should increase numerous numbers of records and it would be very useful to estimate the price for every single room in the future.


We have scraped some quarterly historical prices but still did not use in this project since there were some unreliability issues in the data. It required verification and data cleaning. This historical data can be really useful to visualize the trends for each condo/area (which areas are growing rapidly and/or reaching plateau stage or declining). 


For detail explanation, please refer to the [PDF report](https://github.com/ekapope/2019-01-Web-Scraping-using-selenium-and-bs4/blob/master/20190408_Project_Description.pdf).



