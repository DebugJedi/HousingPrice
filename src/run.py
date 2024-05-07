import pandas as pd
import numpy as np
from HousingPrice.preprocessing import data
from HousingPrice.preprocessing import descriptive
from HousingPrice.preprocessing import CombinedAttributesAdder
from HousingPrice.preprocessing import preprocessing_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import warnings


warnings.filterwarnings('ignore')

# Initial setup and loading the data...
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
data.fetch_housing_data()
df = data.load_housing_data()


# columns =>  ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
#              'total_bedrooms', 'population', 'households', 'median_income', 
#              'median_house_value', 'ocean_proximity']

# bins = [0,1.5, 3.0, 4.5, 6, np.inf]
# labels=[1,2,3,4,5])

# split the data into test and train
cls_split = preprocessing_split(df)
train_df, test_df = cls_split.split()
train_df.shape
test_df.shape

train_x_housing = train_df.drop("median_house_value", axis = 1)
train_y_housing = train_df['median_house_value'].copy()

test_x_housing = test_df.drop("median_house_value", axis =1)
test_y_housing = test_df["median_house_value"].copy()

# preparing the pipeline to automate training and test with new update
numeric = ["float64"]
numeric_columns = list(train_x_housing.select_dtypes(include=numeric ))

train_numeric = train_x_housing[numeric_columns]
train_cat = train_x_housing[["ocean_proximity"]].copy()

train_numeric_col = list(train_numeric)
train_cat_col = list(train_cat)

# Numeric and categorical preprocessing pipeline
numeric_processor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("attribute_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler())
    ])

# numeric_processor['std_scaler'].fit_transform(train_numeric)

cat_processor = Pipeline(
    steps= [
        ("impute_const", SimpleImputer(fill_value="missing", strategy="constant")),
        ("OneHot", OneHotEncoder(handle_unknown="ignore"))
    ])


# combine numeric and categorical processors

preprocessor = ColumnTransformer(
    [
        ("num", numeric_processor, train_numeric_col),
        ("cat", cat_processor, train_cat_col )
    ]
)

# Using custom pipeline combined the preprocessor pipeline with a prediction model (LinearRegression)
pipe = make_pipeline(preprocessor, LinearRegression())

pipe.fit(train_x_housing, train_y_housing)
y_predict = pipe.predict(test_x_housing)

lin_mse = mean_squared_error(test_y_housing, y_predict)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)
# The lin_rmse for the LinerRegression() model came close to: 66924.9521680714

pipe2 = make_pipeline(preprocessor, DecisionTreeRegressor())

pipe2.fit(train_x_housing, train_y_housing)
y2_prediction = pipe2.predict(test_x_housing)

tree_mse = mean_squared_error(test_y_housing, y2_prediction)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)
# The tree_rmse for the decisionTreeRegressor() model came close to: 69741.68005758018
#not a whole lot better...
#let's fine tune the model we have to be able to get best out of what we have..

RF_pipe = Pipeline(
    steps= [("preprocessor", preprocessor), ("regressor", RandomForestRegressor())]
)
RF_pipe.fit(train_x_housing, train_y_housing)
RF_pipe.predict(test_x_housing)

param_grid = [
    {'regressor__n_estimators': [3,10,30],
     'regressor__max_features': ["auto", "sqrt", "log2"],
      'regressor__max_depth': [4,5,6,8] }
]

grid_search = GridSearchCV(RF_pipe, param_grid=param_grid, n_jobs=1)
grid_search.fit(train_x_housing, train_y_housing)
grid_search.best_params_

RF_pipe = Pipeline(
    steps= [("preprocessor", preprocessor), ("regressor", RandomForestRegressor(max_depth=8,
                                                                                max_features='auto',
                                                                                n_estimators=30))]
)
RF_pipe.fit(train_x_housing, train_y_housing)
RF_y_Predict = RF_pipe.predict(test_x_housing)

RF_mse = mean_squared_error(test_y_housing, RF_y_Predict)
RF_rmse = np.sqrt(RF_mse)
print(RF_rmse)
# The RMSE comes to be: 54812.62165596689 better than other models but not the best... 