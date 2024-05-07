"""
This package contains all the modules that deals with loading and preprocessing the data. 
"""
from HousingPrice.preprocessing.data import fetch_housing_data
from HousingPrice.preprocessing.data import load_housing_data
from HousingPrice.preprocessing.descriptive import descriptive_details
from HousingPrice.preprocessing.descriptive import plots_the_data
from HousingPrice.preprocessing.descriptive import scatter_plot
from HousingPrice.preprocessing.transformation import CombinedAttributesAdder
from HousingPrice.preprocessing.data_split import preprocessing_split