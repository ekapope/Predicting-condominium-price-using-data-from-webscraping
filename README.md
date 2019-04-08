# Predicting condominium price using data from webscraping


### 1.	Data set and explanation webscraping process
This project uses Selenium library to firstly obtain all condominiums listed on the https://www.hipflat.com/ website, and extracts information for each page using BeautifulSoup package. Hipflat is one of the biggest property listing website in Thailand. This project is focused on condominium listings in Bangkok, both new and resale. Refer to below links for Python scripts.

[001_Retrieve_all_urls.py](https://github.com/ekapope/2019-01-Web-Scraping-using-selenium-and-bs4/blob/master/001_Retrieve_all_urls.py)

[002_Scrape_info_for_each_condo.py](https://github.com/ekapope/2019-01-Web-Scraping-using-selenium-and-bs4/blob/master/002_Scrape_info_for_each_condo.py)


### 2.	Pre-processing and data cleaning
Check NAs and data types for each column. Perform data manipulation by clean each column using regex, change numbers from strings to numeric, impute missing values, and convert lists of strings into columns. Refer to the link below.

[003_data_cleaning_pred_current_price.py](https://github.com/ekapope/2019-01-Web-Scraping-using-selenium-and-bs4/blob/master/003_data_cleaning_pred_current_price.py)


### 3.	Data scaling and hyperparameter tuning
Robust Scaler  is used in the pipeline before passing through the ML models. It uses a similar method to the Min-Max scaler but it instead uses the interquartile range, rather than the min-max, so that it is robust to outliers. 

[004_1_ML_pred_current_price.py](https://github.com/ekapope/2019-01-Web-Scraping-using-selenium-and-bs4/blob/master/004_1_ML_pred_current_price.py)