import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import helper as hp
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,roc_auc_score
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import iplot
metric_constraints = ['#2ECC71','#34495E','#D0D3D4']

df = pd.read_csv('House_Rent_Dataset.csv')

hp.check_df(df)

df.rename(columns = {'BHK': 'NUM_OF_ROOM',
           'Furnishing Status': 'FURN_STAT',
           'Tenant Preferred': 'TENANT_PREF',
           'Point of Contact': 'CONTACT'}, inplace=True)

##################          EDA           ##################

cat_cols, num_cols, cat_but_car = hp.grab_col_names(df)

for col in cat_cols:
    hp.cat_summary(df, col)

for col in num_cols:
    hp.num_summary(df, col)

for col in cat_cols:
    hp.target_summary_with_cat(df, 'Rent', col)

hp.desc_stats(df)

hp.check_class(df)


####################      FEATURE ENGINEERING       ##################

### OUTLIER ANALYSIS ###

for col in num_cols:
    print(col, hp.check_outlier(df, col))

num_cols = ['Rent', 'Size']

for col in num_cols:
    hp.replace_with_thresholds(df, col)

### Column By Column
df.drop('Area Locality', inplace=True, axis=1)

df['Post_Day'] = df['Posted On'].apply(lambda x: x.split('-')[-1])
df['Post_Month'] = df['Posted On'].apply(lambda x: x.split('-')[1])
df.drop('Posted On', axis=1, inplace=True)

df['Floor'] = df['Floor'].apply(lambda x: x.split(' ')[0])
df['Floor'].replace({'Ground': 0, 'Upper': 2, 'Lower': 1}, inplace=True)
 
df['Area Type'].replace({'Built Area': 'Super Area'}, inplace=True)

df['CONTACT'].replace({'Contact Builder': 'Contact Owner'}, inplace=True)

dff = df.copy()
df = dff.copy()

df.head()
### Creating New Variables

# RATIO
df['Size_Per_Room'] = df['Size'] / (df['NUM_OF_ROOM'] + 1)
df['Size_Per_Bath'] = df['Size'] / (df['Bathroom'] + 1)

df['Num_Plus_Bath'] = df['NUM_OF_ROOM'] + df['Bathroom']
df['Room_Mul_Bath'] = (df['NUM_OF_ROOM'] * 1) + (df['Bathroom'] + 1)


df['CITY_RENT_MEAN'] = df.groupby('City')['Rent'].transform('mean')
df['CITY_RENT_MAX'] = df.groupby('City')['Rent'].transform('max')
df['CITY_RENT_MIN'] = df.groupby('City')['Rent'].transform('min')
df['CITY_SIZE_MEAN'] = df.groupby('City')['Size'].transform('mean')
df['CITY_SIZE_MAX'] = df.groupby('City')['Size'].transform('max')
df['CITY_SIZE_MIN'] = df.groupby('City')['Size'].transform('min')

df.loc[(df['Size'] < 750), 'Area'] = 'Small'
df.loc[(df['Size'] >= 750) & (df['Size'] < 1500), 'Area'] = 'Middle'
df.loc[(df['Size'] >= 1500) & (df['Size'] <= df['Size'].max()), 'Area'] = 'Big'
df.groupby(by = 'Area')['Rent'].mean()


df['FURN_STAT'].replace({'Unfurnished': 0, 'Semi-Furnished': 1, 'Furnished': 2}, inplace=True)
df['Size_Mul_Furn'] = df['Size'] * (df['FURN_STAT'] + 1)
df['Num_Plus_Furn'] = df['NUM_OF_ROOM'] + df['FURN_STAT']
df['Bath_Plus_Furn'] = df['Bathroom'] + df['FURN_STAT']
df.head()

df['Size_Mean'] = df['CITY_SIZE_MEAN'] / df['Size']
###################                   ENCODING VE SCALING                ##########################

cat_cols, num_cols, cat_but_car = hp.grab_col_names(df)
num_cols

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

cat_cols

df = hp.one_hot_encoder(df,cat_cols, drop_first=True)

df.head()

df[['Post_Day', 'Post_Month']] = rs.fit_transform(df[['Post_Day', 'Post_Month']])

#####################               MODELING                ###################

y = df['Rent']
X = df.drop(['Rent'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=42)

models = [('LR', LinearRegression()),
        ("Ridge", Ridge()),
        ("Lasso", Lasso()),
        ("ElasticNet", ElasticNet()),
        ('KNN', KNeighborsRegressor()),
        ('CART', DecisionTreeRegressor()),
        ('RF', RandomForestRegressor()),
        ('SVR', SVR()),
        ('GBM', GradientBoostingRegressor()),
        ("XGBoost", XGBRegressor(objective='reg:squarederror')),
        ("LightGBM", LGBMRegressor()),
        ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
     rmse = np.mean(np.sqrt(-cross_val_score(regressor, X_train, y_train, cv=7, scoring="neg_mean_squared_error")))
     print(f"RMSE: {round(rmse, 4)} ({name}) ")


catb_model = CatBoostRegressor().fit(X_train,y_train)


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(catb_model, X)