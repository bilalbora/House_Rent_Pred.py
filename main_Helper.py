import pandas as pd
import numpy as np
import plotly as plotly
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
colors = ['#494BD3', '#E28AE2', '#F1F481', '#79DB80', '#DF5F5F',
              '#69DADE', '#C2E37D', '#E26580', '#D39F49', '#B96FE3']
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)


#############################
# KEŞİFÇİ VERİ ANALİZİ (EDA)
#############################

# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)


###########################
#       GENEL RESİM        #
# #########################

def check_df(dataframe, head=5):

    pd.set_option('display.max_columns', None)   #*
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 170)

    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Info #####################")  #*
    print(dataframe.info())

    print("##################### Columns #####################")  #*
    print(dataframe.columns)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Describe #####################")
    print(dataframe.describe().T)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]


    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


######## KATEGORİK DEĞİŞKEN ANALİZİ VE GÖRSELİ ############

def cat_summary(dataframe, col_name, plot=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

def cat_summary_with_graph(dataframe, col_name):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Countplot', 'Percentages'),
                        specs=[[{"type": "xy"}, {'type': 'domain'}]])

    fig.add_trace(go.Bar(y=dataframe[col_name].value_counts().values.tolist(),
                         x=[str(i) for i in dataframe[col_name].value_counts().index],
                         text=dataframe[col_name].value_counts().values.tolist(),
                         textfont=dict(size=15),
                         name=col_name,
                         textposition='auto',
                         showlegend=False,
                         marker=dict(color=colors,
                                     line=dict(color='#DBE6EC',
                                               width=1))),
                  row=1, col=1)

    fig.add_trace(go.Pie(labels=dataframe[col_name].value_counts().keys(),
                         values=dataframe[col_name].value_counts().values,
                         textfont=dict(size=20),
                         textposition='auto',
                         showlegend=False,
                         name=col_name,
                         marker=dict(colors=colors)),
                  row=1, col=2)

    fig.update_layout(title={'text': col_name,
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      template='plotly_white')

    iplot(fig)

############### SAYISAL DEĞİŞKEN ANALİZİ VE GÖRSELİ ##################

def num_summary(dataframe, numerical_col, plot=False):

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

def num_summary_with_graph(dataframe, col_name):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Quantiles', 'Distribution'))

    fig.add_trace(go.Box(y=dataframe[col_name],
                         name=str(col_name),
                         showlegend=False,
                         marker_color=colors[1]),
                  row=1, col=1)

    fig.add_trace(go.Histogram(x=dataframe[col_name],
                               xbins=dict(start=dataframe[col_name].min(),
                                          end=dataframe[col_name].max()),
                               showlegend=False,
                               name=str(col_name),
                               marker=dict(color=colors[0],
                                           line=dict(color='#DBE6EC',
                                                     width=1))),
                  row=1, col=2)

    fig.update_layout(title={'text': col_name,
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      template='plotly_white')

    iplot(fig)


def num_plot(data, cat_length=16, remove=["Id"], hist_bins=12, figsize=(20, 4)):
    num_cols = [col for col in data.columns if data[col].dtypes != "O"
                and len(data[col].unique()) >= cat_length]

    if len(remove) > 0:
        num_cols = list(set(num_cols).difference(remove))

    for i in num_cols:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        data.hist(str(i), bins=hist_bins, ax=axes[0])
        data.boxplot(str(i), ax=axes[1], vert=False);
        try:
            sns.kdeplot(np.array(data[str(i)]))
        except:
            ValueError

        axes[1].set_yticklabels([])
        axes[1].set_yticks([])
        axes[0].set_title(i + " | Histogram")
        axes[1].set_title(i + " | Boxplot")
        axes[2].set_title(i + " | Density")
        plt.show()

#################       HEDEF DEĞİŞKEN ANALİZİ      ################

def target_summary_with_cat(dataframe, target, categorical_col, plot=False):

    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby([categorical_col])[target].mean(),
                        "TARGET_SUM" : dataframe.groupby([categorical_col])[target].sum()}), end="\n\n\n")

##################      KORELASYON ANALİZİ      #######################

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

# UNİQUE SAYISI ÖĞRENME

def check_class(dataframe):
    nunique_df = pd.DataFrame({'Variable': dataframe.columns,
                               'Classes': [dataframe[i].nunique() \
                                           for i in dataframe.columns]})

    nunique_df = nunique_df.sort_values('Classes', ascending=False)
    nunique_df = nunique_df.reset_index(drop = True)
    return nunique_df


#############################
#   FEATURE ENGINEERING     #
#############################

# 1. Outliers (Aykırı Değerler)
# 2. Missing Values (Eksik Değerler)
# 3. Feature Extraction (Özellik Çıkarımı)
# 4. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
# 5. Feature Scaling (Özellik Ölçeklendirme)

#############################################
# 1. Outliers (Aykırı Değerler)
#############################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# for col in num_cols:
#     print(col, check_outlier(df, col))
#
# for col in num_cols:
#     if check_outlier(df, col):
#         replace_with_thresholds(df, col)

#############################################
# 2. Missing Values (Eksik Değerler)
#############################################

def missing_values(data, plot=False, target="SalePrice"):
    mst = pd.DataFrame(
        {"Num_Missing": df.isnull().sum(), "Missing_Ratio": df.isnull().sum() / df.shape[0]}).sort_values("Num_Missing",
                                                                                                          ascending=False)
    mst["DataTypes"] = df[mst.index].dtypes.values
    mst = mst[mst.Num_Missing > 0].reset_index().rename({"index": "Feature"}, axis=1)
    mst = mst[mst.Feature != target]

    print("Number of Variables include Missing Values:", mst.shape[0], "\n")

    if mst[mst.Missing_Ratio > 0.99].shape[0] > 0:
        print("Full Missing Variables:", mst[mst.Missing_Ratio > 0.99].Feature.tolist())
        data.drop(mst[mst.Missing_Ratio > 0.99].Feature.tolist(), axis=1, inplace=True)

        print("Full missing variables are deleted!", "\n")

    if plot:
        plt.figure(figsize=(25, 8))
        p = sns.barplot(mst.Feature, mst.Missing_Ratio)
        for rotate in p.get_xticklabels():
            rotate.set_rotation(90)

    print(mst, "\n")

#############################################
# 4. One-Hot Encoding
#############################################

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

#############################################
# 5. Feature Scaling (Özellik Ölçeklendirme)
#############################################

##  SCALING İŞLEMLERİ YAPILACAK



#############################################
# Regresyon Base Models
#############################################

# y = df["Salary"]
# X = df.drop(["Salary"], axis=1)
#
# models = [('LR', LinearRegression()),
#           ("Ridge", Ridge()),
#           ("Lasso", Lasso()),
#           ("ElasticNet", ElasticNet()),
#           ('KNN', KNeighborsRegressor()),
#           ('CART', DecisionTreeRegressor()),
#           ('RF', RandomForestRegressor()),
#           ('SVR', SVR()),
#           ('GBM', GradientBoostingRegressor()),
#           ("XGBoost", XGBRegressor(objective='reg:squarederror')),
#           ("LightGBM", LGBMRegressor()),
#           ("CatBoost", CatBoostRegressor(verbose=False))]
#
#
# for name, regressor in models:
#     rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
#     print(f"RMSE: {round(rmse, 4)} ({name}) ")



############    MODEL ÇAĞIRMA  (CLASSİFİCATİON)   #########

    # classifiers = [('LR', LogisticRegression()),
    #            ('KNN', KNeighborsClassifier()),
    #            ("SVC", SVC()),
    #            ("CART", DecisionTreeClassifier()),
    #            ("RF", RandomForestClassifier()),
    #            ('Adaboost', AdaBoostClassifier()),
    #            ('GBM', GradientBoostingClassifier())
    #            ]
    #
    # for name, classifier in classifiers:
    #     cv_results = cross_validate(classifier, X, y, cv=3, scoring=["roc_auc"])
    #     print(f"AUC: {round(cv_results['test_roc_auc'].mean(),7)} ({name}) ")




#######     DEĞİŞKEN ÖNEMİ      ##########

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

# plot_importance(rf_final, X)
# plot_importance(gbm_final, X)
# plot_importance(lgbm_final, X)
# plot_importance(catboost_final, X)


######     HYPERMETRE OPTİMİZASYONU  (MODEL İÇİN EN İYİ PARAMETRELER)   ########

##############################################
#
# rf_model = RandomForestRegressor(random_state=17)
#
# rf_params = {"max_depth": [5, 8, 15, None],
#              "max_features": [5, 7, "auto"],
#              "min_samples_split": [8, 15, 20],
#              "n_estimators": [200, 500]}
#
# rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
# rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)
# rmse = np.mean(np.sqrt(-cross_val_score(rf_final, X, y, cv=10, scoring="neg_mean_squared_error")))
# rmse
#
# ################################################
# # GBM Model
# ################################################
#
# gbm_model = GradientBoostingRegressor(random_state=17)
#
# gbm_params = {"learning_rate": [0.01, 0.1],
#               "max_depth": [3, 8],
#               "n_estimators": [500, 1000],
#               "subsample": [1, 0.5, 0.7]}
#
# gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
# gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)
# rmse = np.mean(np.sqrt(-cross_val_score(gbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))
# rmse
#
# ################################################
# # LightGBM
# ################################################
#
# lgbm_model = LGBMRegressor(random_state=17)
#
# lgbm_params = {"learning_rate": [0.01, 0.1],
#                 "n_estimators": [300, 500],
#                 "colsample_bytree": [0.7, 1]}
#
# lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
# lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)
# rmse = np.mean(np.sqrt(-cross_val_score(lgbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))
# rmse
#
# ################################################
# # CatBoost
# ################################################
#
# catboost_model = CatBoostRegressor(random_state=17, verbose=False)
#
# catboost_params = {"iterations": [200, 500],
#                    "learning_rate": [0.01, 0.1],
#                    "depth": [3, 6]}
#
# catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
# catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)
# rmse = np.mean(np.sqrt(-cross_val_score(catboost_final, X, y, cv=10, scoring="neg_mean_squared_error")))


######################
# MODEL ANALİZİ
#####################



################################################
# Analyzing Model Complexity with Learning Curves
#################################################

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()

# rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
#                  ["max_features", [3, 5, 7, "auto"]],
#                  ["min_samples_split", [2, 5, 8, 15, 20]],
#                  ["n_estimators", [10, 50, 100, 200, 500]]]
#
#
# rf_model = RandomForestRegressor(random_state=17)
#
# for i in range(len(rf_val_params)):
#     val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1],scoring="neg_mean_absolute_error")
#
# rf_val_params[0][1]
















