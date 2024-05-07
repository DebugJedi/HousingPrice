from HousingPrice.preprocessing import data
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


def descriptive_details(df = data.load_housing_data()):
    print(df.head())
    print(32*"#")
    print("Info....")
    print(df.info())
    print(32*"#")
    print(df["ocean_proximity"].value_counts())
    print(32*"#")
    print("Let's look at the numerical summary...")
    print(df.describe())
    print(32*"#")

def plots_the_data(df = data.load_housing_data(), bins=50, figsize=(20,15)):
    df.hist(bins=bins, figsize=figsize)
    plt.show()

def scatter_plot(df = data.load_housing_data(), figsize= (12, 8)):
    numerics = ['int64', 'float64']
    attributes = list(df.select_dtypes(include=numerics))
    scatter_matrix(df[attributes], figsize=figsize)
    plt.show()

