# Import necessary libraries
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_validate, GridSearchCV

# Basic DataFrame Information
def check_df(dataframe, head=5):
    """
    Display basic information about the dataframe, including shape, types, head, tail,
    missing values, and quantiles.
    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99]).T)


# Column Classification by Data Type
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Returns the names of categorical, numerical, and cardinal categorical variables.
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, cat_but_car, num_cols, num_but_cat


# Summary of Categorical Variables
def cat_summary(dataframe, col_name, plot=False):
    """
    Displays the frequency and ratio of a categorical variable.
    Optionally plots a count plot if plot=True.
    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

    print("#####################################")


# Summary of Numerical Variables
def num_summary(dataframe, numerical_col, plot=False):
    """
    Displays summary statistics of a numerical variable.
    Optionally plots a histogram if plot=True.
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=False)

    print("#####################################")


# Check Skewness of Numerical Variables
def check_skew(dataframe, column):
    """
    Checks the skewness of a numerical variable and visualizes its distribution.
    """
    skew = stats.skew(dataframe[column])
    skewtest = stats.skewtest(dataframe[column])
    plt.title('Distribution of ' + column)
    sns.displot(dataframe, x=column, color="g", kde=True)
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    plt.show()


# Label Encoding for Binary Categorical Variables
def label_encoder(dataframe, binary_col):
    """
    Encodes binary categorical variables into numerical values.
    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


# Rare Category Analysis
def rare_analyser(dataframe, target, cat_cols):
    """
    Analyzes the distribution of rare categories in categorical variables and their relationship with the target variable.
    """
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


# Rare Category Encoding
def rare_encoder(dataframe, rare_perc):
    """
    Combines rare categories in categorical variables into a single 'Rare' category based on a specified threshold.
    """
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


# One-Hot Encoding for Categorical Variables
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """
    Applies one-hot encoding to categorical variables.
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# Outlier Thresholds Calculation
def outlier_thresholds(dataframe, variable, low_quantile=0.05, up_quantile=0.95):
    """
    Calculates the thresholds for detecting outliers based on quantiles.
    """
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


# Check for Outliers
def check_outlier(dataframe, col_name):
    """
    Checks if a column contains outliers based on calculated thresholds.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    outlier_present = (dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)
    return outlier_present.any()


# Grab Outliers from DataFrame
def grab_outliers(dataframe, col_name, index=False):
    """
    Grabs and returns the outliers in a DataFrame. Optionally returns their indices.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    outliers = dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)]

    if outliers.shape[0] > 0:
        if index:
            return outliers.index
        else:
            return outliers
    else:
        print(f"No outliers found in {col_name}")
        return pd.DataFrame()


# Target Variable Analysis with Numerical Variables
def target_summary_with_num(dataframe, target, numerical_col):
    """
    Analyzes the relationship between a numerical variable and the target variable.
    """
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


# Target Variable Analysis with Categorical Variables
def target_summary_with_cat(dataframe, target, categorical_col):
    """
    Analyzes the relationship between a categorical variable and the target variable.
    """
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")


# Plot Feature Importance
def plot_importance(model, X, num=len(X)):
    """
    Plots the importance of features in a model.
    """
    importances = model.feature_importances_
    feature_names = X.columns.tolist()
    feature_imp = pd.DataFrame({'Feature': feature_names, 'Value': importances})
    top_features = feature_imp.sort_values(by="Value", ascending=False).head(num)
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=top_features)
    plt.title('Top {} Important Features'.format(num))
    plt.show()

# Compare Base Models to identify the best model for our dataset
def base_models(X, y, scoring="accuracy"):
    """
    Compares different baseline models using cross-validation and prints the average performance metric.

    Parameters:
    X : DataFrame or array-like
        Features dataset.
    y : Series or array-like
        Target variable.
    scoring : str, optional
        Scoring metric to use for evaluation, default is "accuracy".

    Returns:
    None
    """
    print("Base Models...")
    models = [('LR', LogisticRegression()),
              ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier()),
              ('RF', RandomForestClassifier()),
              ('GBM', GradientBoostingClassifier()),
              ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')),
              ("LightGBM", LGBMClassifier()),
              ("CatBoost", CatBoostClassifier(verbose=False))]

    for name, classifier in models:
        cv_results = np.mean(cross_val_score(classifier, X, y, cv=5, scoring=scoring))
        print(f"{scoring}: {round(cv_results, 4)} ({name})")


# Hyperparameter Optimization for Models
def hyperparameter_optimization(X, y, classifiers, cv=5, scoring="roc_auc"):
    """
    Performs hyperparameter optimization using GridSearchCV and evaluates the performance of the models.

    Parameters:
    X : DataFrame or array-like
        Features dataset.
    y : Series or array-like
        Target variable.
    classifiers : list of tuples
        List where each tuple contains ('model_name', model_instance, param_grid).
    cv : int, optional
        Number of cross-validation folds, default is 5.
    scoring : str, optional
        Scoring metric to use for evaluation, default is "roc_auc".

    Returns:
    best_models : dict
        Dictionary of models with their optimized hyperparameters.
    """
    print("Hyperparameter Optimization...")
    best_models = {}

    for name, classifier, params in classifiers:
        print(f"############   {name}   ##############")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring}(Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring}(After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

        best_models[name] = final_model

    return best_models
