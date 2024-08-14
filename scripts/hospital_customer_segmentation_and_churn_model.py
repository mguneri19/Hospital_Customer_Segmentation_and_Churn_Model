# Import necessary libraries
import numpy as np  # For numerical operations and array manipulation
import pandas as pd  # For data manipulation and analysis
import datetime as dt  # For handling date and time operations
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
import seaborn as sns  # For making statistical graphics more informative

# Import warning control and preprocessing tools
import warnings  # For controlling warning messages
from sklearn.preprocessing import MinMaxScaler  # For feature scaling
from sklearn.cluster import KMeans  # For performing K-means clustering
from yellowbrick.cluster import KElbowVisualizer  # For determining the optimal number of clusters using the elbow method
from sklearn.exceptions import ConvergenceWarning  # To handle specific convergence warnings in models

# Import tools for data preprocessing and model evaluation
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables into numeric labels
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV  # For splitting data and performing cross-validation
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score  # For evaluating model performance

# Import machine learning models
from sklearn.linear_model import LogisticRegression  # Logistic regression model
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors classifier
from sklearn.tree import DecisionTreeClassifier  # Decision Tree classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Random Forest and Gradient Boosting models
from xgboost import XGBClassifier  # XGBoost model, a powerful gradient boosting algorithm
from lightgbm import LGBMClassifier  # LightGBM model, another gradient boosting framework
from catboost import CatBoostClassifier  # CatBoost model, gradient boosting model optimized for categorical data
from xgboost import plot_importance  # For plotting feature importance in XGBoost

# Import necessary functions from utils library
from utils import check_df, grab_col_names, cat_summary, num_summary, check_skew, label_encoder # DataFrame Overview Functions
from utils import rare_analyser, rare_encoder, one_hot_encoder, outlier_thresholds, check_outlier # Rare and Outlier Handling Functions
from utils import grab_outliers, target_summary_with_num, target_summary_with_cat, plot_importance # Outlier Detection and Feature Importance Functions
from utils import base_models, hyperparameter_optimization # Functions comparing different baseline models and making hyperparameter optimization

# Import joblib for saving and loading machine learning models
import joblib  # For saving models as .pkl files and loading them later

# Ignore specific warnings to keep the output clean
warnings.simplefilter(action='ignore', category=FutureWarning)  # Ignore FutureWarning messages
warnings.simplefilter("ignore", category=ConvergenceWarning)  # Ignore ConvergenceWarning messages

# Set Pandas display options for better output readability
pd.set_option('display.max_columns', None)  # Show all columns when displaying DataFrames
pd.set_option('display.max_rows', None)  # Show all rows when displaying DataFrames
pd.set_option('display.width', None)  # Adjust the display width for DataFrames
pd.set_option('display.float_format', lambda x: '%.3f' % x)  # Format float numbers to three decimal places


# Load data
df_ = pd.read_excel(r"data/2023_Ver.xlsx")  # Add the path to your data file
df = df_.copy() # Copy the data
df.head()  # Display the first few rows to understand the structure of the data
# Check the shape and number of unique values in each column
print(df.shape)
df.nunique()

##################   Data Cleaning & Preprocessing   ######################################################################
# The dataset contains unique values and may have duplicate customers under different categories.
# To identify unique patients, we'll create a new 'Birleşik' feature by combining several relevant columns.
df['Birleşik'] = df['Cinsiyet'] + df['Yaş'].astype(str) + df['Şehir'] + df['İlçe'] + df['Yerli-Yabancı Hasta'] + df['Sigorta Türü(KURUM)'] + df['Ayaktan /Yatan Hasta']

# Assign a unique customer ID to each unique combination in the 'Birleşik' column
df['Müşteri_ID'] = df.groupby('Birleşik').ngroup()
df["Müşteri_ID"].nunique()  # There are 146,714 unique customers in the dataset
df["Müşteri ID"].nunique()  # The original 'Müşteri ID' column had 374,672 rows (total customers who visited the hospital during the period


# General overview of the dataset
check_df(df)

# Dropping columns and handling missing values
df.drop(["Sigorta Türü(EK KURUM)"], axis=1, inplace=True)  # Drop the 'Sigorta Türü(EK KURUM)' column due to excessive missing values
df.dropna(subset=["Şehir"], inplace=True)  # Drop rows with missing values in the 'Şehir' column due to few missing values
df.dropna(subset=["İlçe"], inplace=True)  # Drop rows with missing values in the 'İlçe' column due to few missing values
df.drop(["Müşteri ID", "Birleşik"], axis=1, inplace=True)  # Drop unnecessary columns now that we have a unique 'Müşteri_ID'

# Renaming some columns for simplicity and readability
df.rename(columns={"Sigorta Türü(KURUM)": "Sigorta Türü"}, inplace=True)
df.rename(columns={"Ayaktan /Yatan Hasta": "Hasta Türü"}, inplace=True)

####################################################################################################################################


##################   Exploratory Data Analysis- 1    ###############################################################################
# Function to identify categorical and numerical columns
cat_cols, cat_but_car, num_cols = grab_col_names(df)

# Review unique value counts to identify potential feature engineering opportunities
df.nunique()

# Consolidate categories for several columns to reduce cardinality
df["Yerli-Yabancı Hasta"] = np.where(df["Yerli-Yabancı Hasta"] == "TÜRKİYE CUMHURİYETİ", "Yerli", "Yabancı")  # Consolidate 'Yerli-Yabancı Hasta' into two categories

df["Sigorta Türü"] = df["Sigorta Türü"].apply(lambda x: x if x in ["SSK", "BAĞKUR", "EMEKLİ SANDIĞI", "ÜCRETLİ HASTA"] else "ÖZEL SAĞLIK")  # Consolidate 'Sigorta Türü' into five categories

df["Şehir"] = np.where(df["Şehir"] == "İSTANBUL", "İstanbul", "Şehir Dışı")  # Consolidate 'Şehir' into two categories

# Consolidate 'İlçe' into fewer categories based on frequency
conditions = [
    df["İlçe"] == "ŞİŞLİ",
    df["İlçe"] == "KAĞITHANE",
    df["İlçe"] == "BEYOĞLU",
    df["İlçe"] == "İSTANBUL",
    df["İlçe"] == "SARIYER",
    df["İlçe"] == "EYÜP",
    df["İlçe"] == "BEŞİKTAŞ"
]
choices = ["ŞİŞLİ", "KAĞITHANE", "BEYOĞLU", "İSTANBUL", "SARIYER", "EYÜP", "BEŞİKTAŞ"]
df["İlçe"] = np.select(conditions, choices, default="DİĞER")


# Create a new feature as column 'Bölge' based on 'İlçe'
conditions2 = [
    df["İlçe"].isin(["ŞİŞLİ", "KAĞITHANE", "BEYOĞLU", "İSTANBUL"]),
    df["İlçe"].isin(["SARIYER", "EYÜP", "BEŞİKTAŞ"])
]
choices2 = ['1. Bölge', '2. Bölge']
df["Bölge"] = np.select(conditions2, choices2, default="3. Bölge")



# Simplify 'Hizmet aldığı Branş' into three categories based on frequency
brans_sayilari = df["Hizmet aldığı Branş"].value_counts()
dusuk_yogunluklu_branslar = brans_sayilari[brans_sayilari < 1000].index
df.loc[df["Hizmet aldığı Branş"].isin(dusuk_yogunluklu_branslar), "Branş Yoğunluğu"] = "Düşük-Yoğun Branş"
orta_yogunluklu_branslar = brans_sayilari[(brans_sayilari >= 1000) & (brans_sayilari < 10000)].index
df.loc[df["Hizmet aldığı Branş"].isin(orta_yogunluklu_branslar), "Branş Yoğunluğu"] = "Orta-Yoğun Branş"
yuksek_yogunluklu_branslar = brans_sayilari[(brans_sayilari >= 10000)].index
df.loc[df["Hizmet aldığı Branş"].isin(yuksek_yogunluklu_branslar), "Branş Yoğunluğu"] = "Yüksek-Yoğun Branş"

# Create age groups based on domain knowledge
df["Yaş Grubu"] = pd.cut(x=df["Yaş"],
                         bins=[0, 2, 12, 18, 40, 70, 85, 105],
                         labels=["Bebek", "Çocuk", "Ergen","Genç", "Yetişkin", "Orta Düzey Yaşlı", "İleri Düzey Yaşlı"])


# Simplify 'Tedavi Adı' by keeping only the first part before the comma
df["Tedavi Adı"] = df["Tedavi Adı"].str.split(',').str[0]
df["Tedavi Adı"].nunique()  # The number of unique treatment types has decreased

# Classify 'Tedavi Adı' into four categories based on frequency
tedavi = df["Tedavi Adı"].value_counts().sort_values(ascending=False)
ilk_40_tedavi = df["Tedavi Adı"].value_counts().head(40).index  # First 40 most common treatments
tedavi_40_100 = tedavi[40:100].index  # Treatments ranked 40 to 100
tedavi_100_200 = tedavi[100:200].index  # Treatments ranked 100 to 200
diger_tedaviler = tedavi[200:2228].index  # Treatments ranked 200 to 2228

# Classify treatments into four categories: 'En Yaygın', 'Sık Yapılan', 'Orta Sıklıkta', 'Nadir Yapılan'
df.loc[df["Tedavi Adı"].isin(ilk_40_tedavi), "Tedavi Sınıflandırması"] = "En Yaygın"
df.loc[df["Tedavi Adı"].isin(tedavi_40_100), "Tedavi Sınıflandırması"] = "Sık Yapılan"
df.loc[df["Tedavi Adı"].isin(tedavi_100_200), "Tedavi Sınıflandırması"] = "Orta Sıklıkta"
df.loc[df["Tedavi Adı"].isin(diger_tedaviler), "Tedavi Sınıflandırması"] = "Nadir Yapılan"


# Re-run the function to get updated lists of categorical, cardinal, and numerical columns
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

df.nunique()

# Analyze categorical variables
# Summarize each categorical variable
for col in cat_cols:
    cat_summary(df, col)

# Analyze numerical variables
# Summarize each numerical variable
for col in num_cols:
    num_summary(df, col)

# Identify and handle negative 'Toplam Harcama' values
df[df["Toplam Harcama"] < 0].value_counts()  # Check for negative values
df = df[df["Toplam Harcama"] >= 0]  # Remove rows with negative 'Toplam Harcama' values

####################################################################################################################################

##################   CRM Analysis    ###############################################################################

##################  a- RFML Model    ###############################################################################

# Identify top 10 customers by total spending
df.groupby("Müşteri_ID").agg({"Toplam Harcama": "sum"}).sort_values("Toplam Harcama", ascending=False).head(10)

# Define the analysis date as two days after the last transaction date in the dataset
df["İşlem Tarihi"].min()  # Earliest transaction date: 2023-01-01
df["İşlem Tarihi"].max()  # Latest transaction date: 2024-02-13
analysis_date = dt.datetime(2024, 2, 15)


# Calculate RFM (Recency, Frequency, Monetary) and Length metrics for each customer
rfml = df.groupby("Müşteri_ID").agg(
    Recency=("İşlem Tarihi", lambda x: (analysis_date - x.max()).days),  # Days since last transaction
    Frequency=("Müşteri_ID", "count"),  # Number of transactions
    Monetary=("Toplam Harcama", "sum"),  # Total spending
    Length=("İşlem Tarihi", lambda x: (analysis_date - x.min()).days)  # Duration of customer relationship
)

# Sort customers by each metric to identify key patterns
rfml.sort_values(by="Frequency", ascending=False).head(10)  # Top 10 by Frequency
rfml.sort_values(by="Recency", ascending=True).head(10)  # Top 10 by Recency (most recent customers)
rfml.sort_values(by="Monetary", ascending=False).head(10)  # Top 10 by Monetary (highest spenders)
rfml.sort_values(by="Length", ascending=False).head(10)  # Top 10 by Length (longest relationships)

# Summary statistics for RFML metrics
print(rfml.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
print(rfml.shape)  # There are 146,714 unique customers


# To better segment customers, we can use K-Means clustering. First, let's check for skewness in the data.

# Check for skewness in the distribution of each RFM metric
from scipy import stats

plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(rfml, "Recency")
plt.subplot(6, 1, 2)
check_skew(rfml, "Frequency")
plt.subplot(6, 1, 3)
check_skew(rfml, "Monetary")
plt.subplot(6, 1, 4)
check_skew(rfml, "Length")

plt.tight_layout()
plt.savefig("before_transform.png", format="png", dpi=1000)
plt.show()

# Apply log transformation to normalize the data
rfml["Recency"] = np.log1p(rfml["Recency"])
rfml["Frequency"] = np.log1p(rfml["Frequency"])
rfml["Monetary"] = np.log1p(rfml["Monetary"])
rfml["Length"] = np.log1p(rfml["Length"])
rfml.head()

# Scale the RFML metrics using MinMaxScaler
sc = MinMaxScaler((0, 1))
rfml_scaled = sc.fit_transform(rfml)
rfml = pd.DataFrame(rfml_scaled, columns=rfml.columns)


##################  b- K-Means Clustering Analysis   ###############################################################################

# Determine the optimal number of clusters using the Elbow method
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(rfml)
elbow.show()  # The optimal number of clusters is 6

# Perform K-Means clustering with the identified optimal number of clusters
k_means = KMeans(n_clusters=6, random_state=42).fit(rfml)
segments = k_means.labels_  # Assign cluster labels to each customer
print(segments)  # Display the cluster assignments for each customer

# Reverse the scaling to return to the original RFML values
rfml_new = sc.inverse_transform(rfml)
rfml_new = pd.DataFrame(rfml_new, columns=rfml.columns)
rfml_new.head()

# Add the cluster labels to the DataFrame
final_rfml = rfml_new
final_rfml["segment"] = segments + 1  # Shift segment labels to start from 1 instead of 0
final_rfml.head()


##################  c- Segment Evaluation   ###############################################################################


# Reset the index to have a clean DataFrame
final_rfml = final_rfml.reset_index()
print(final_rfml.columns)
final_rfml.rename(columns={"index": "Müşteri_ID"}, inplace=True)  # Rename index to 'Müşteri_ID'

# Analyze each segment statistically
final_rfml.groupby("segment").agg({"Recency": ["mean", "min", "max"],
                                   "Frequency": ["mean", "min", "max"],
                                   "Monetary": ["mean", "min", "max"],
                                   "Length": ["mean", "min", "max"]})

# Extract the IDs of customers in the most valuable segment (Segment 1)
segment1 = final_rfml[final_rfml["segment"] == 1]
segment1.value_counts().sum()  # Number of customer in segment 1
segment1_ids = segment1[["Müşteri_ID"]]  # Extract only the customer IDs
segment1_ids.to_csv('segment1_ids.csv', index=False)  # Save the IDs to a CSV file
print(segment1.columns)


##################  d- Identifying Churn-Prone Segments   ###############################################################################

# Calculate the proportion of each segment in the total customer base
total_entries = len(final_rfml)
segment_ratios = {}

for segment in range(1, 7):
        segment_data = final_rfml[final_rfml["segment"] == segment]
        segment_ratios[segment] = len(segment_data) / total_entries

for segment, ratio in segment_ratios.items():
    print(f"Segment {segment}: %{ratio * 100:.2f}")


# Based on this analysis and domain knowledge, the segments with the highest churn risk are:
# Segment 2, Segment 3, and Segment 6, accounting for 51.9% of the customers.
# Segments with higher loyalty are: Segment 1, Segment 4, and Segment 5, accounting for 48.11% of the customers.

# Add segment information to the main dataset
new_final_rfml = final_rfml.drop(columns=["Frequency", "Monetary", "Length", "Recency"])
new_final_rfml.head()

df_merged = pd.merge(df, new_final_rfml, on="Müşteri_ID", how='left')  # Merge the segment information back into the main dataset
df_merged.head()

# Assign a churn flag based on the segments. Segments 2, 3, and 6 are churn-prone.
df_merged["Churn"] = df_merged["segment"].apply(lambda x: 1 if x in [2, 3, 6] else 0)

df_merged.to_csv("df_merged.csv", index=False)  # Save the merged dataset to a CSV file to use for CRM Analysis in Power BI Dashboard


#####################################################################################################################################

##################   Exploratory Data Analysis- 2    ###############################################################################

# Re-check the overall structure and completeness of the merged dataset
check_df(df_merged)

# Identify the updated lists of categorical and numerical columns
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df_merged)

# Categorical variable analysis
cat_cols = [col for col in cat_cols if col not in ["Churn", "segment"]]
for col in cat_cols:
    cat_summary(df_merged, col)

# Numerical variable analysis
num_cols = [col for col in num_cols if col not in ["İşlem Tarihi", "Müşteri_ID"]]
for col in num_cols:
    num_summary(df_merged, col, plot=False)

# Numerical variables vs target variable analysis (Churn)
for col in num_cols:
    target_summary_with_num(df_merged, "Churn", col)

# Categorical variables vs target variable analysis (Churn)
for col in cat_cols:
    target_summary_with_cat(df_merged, "Churn", col)

# Correlation analysis for numerical variables
df_merged[num_cols].corr()

# Plot the correlation matrix
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df_merged[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# The correlation analysis does not show any strong correlations between numerical variables.

# We've already performed some feature engineering based on domain knowledge. Let's check for any remaining opportunities.

# MISSING VALUE ANALYSIS
# We've already checked for missing values in the general overview, and there were none.

# OUTLIER ANALYSIS
# Check for outliers in numerical variables
for col in num_cols:
    print(col, check_outlier(df_merged, col))  # Outliers are found in the 'Toplam Harcama' column

# Count the number of outliers in the 'Toplam Harcama' column
outlier_data = grab_outliers(df_merged, 'Toplam Harcama')
print(f"Number of outliers in Toplam Harcama: {outlier_data.shape[0]}")  # There are 7,516 outliers

# Given the small percentage of outliers, we choose to ignore them, especially since we plan to use tree-based methods where outliers are less impactful.

############################################################################################################################

##################   Feature Engineering / Extraction     ###############################################################################

# FEATURE EXTRACTION
# Review unique values in each column to identify opportunities for new features
for column in df_merged.columns:
    unique_values = df_merged[column].unique()
    print(f"{column} column unique values: {unique_values}")

# Based on our previous analysis, let's create new features by combining significant categories.

# Create a new feature based on gender and insurance type
df_merged.loc[(df_merged["Cinsiyet"] == "ERKEK") & (df_merged["Sigorta Türü"] == "ÖZEL SAĞLIK"), "NEW_Sigorta"] = "OzelSigortaliErkek"
df_merged.loc[(df_merged["Cinsiyet"] == "KADIN") & (df_merged["Sigorta Türü"] == "ÖZEL SAĞLIK"), "NEW_Sigorta"] = "OzelSigortaliKadin"
df_merged.loc[df_merged["Sigorta Türü"] != "ÖZEL SAĞLIK", "NEW_Sigorta"] = "OzelSigortaliDegil"

# Create a new feature based on patient type and communication channel
df_merged.loc[(df_merged["Hasta Türü"] == "Klinik") & (df_merged["İletişim Kanalı"] == "RANDEVUSUZ"), "NEW_Hasta_Turu"] = "RandevusuzKlinikHasta"
df_merged.loc[(df_merged["Hasta Türü"] == "Klinik") & (df_merged["İletişim Kanalı"] == "RANDEVU"), "NEW_Hasta_Turu"] = "RandevuluKlinikHasta"
df_merged.loc[df_merged["Hasta Türü"] != "Klinik", "NEW_Hasta_Turu"] = "KlinikOlmayanHasta"

df_merged.head()
print(df_merged.shape)  # The dataset now contains 21 variables

############################################################################################################################


##################   Encoding    ###########################################################################################

# Separate categorical and numerical columns for encoding
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df_merged)

# Back up the dataset before encoding
df_merged1 = df_merged.copy()

# Label Encoding for binary categorical variables
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
print(binary_cols)  # Binary columns: ['Cinsiyet', 'Şehir', 'Yerli-Yabancı Hasta', 'İletişim Kanalı']

# Apply label encoding to binary columns
for col in binary_cols:
    df_merged = label_encoder(df_merged, col)

# Rare Encoding
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in "Churn"]
print(cat_cols)
# ['İlçe', 'Sigorta Türü', 'Hasta Türü', 'Bölge', 'Branş Yoğunluğu', 'Tedavi Sınıflandırması', 'NEW_Sigorta', 'NEW_Hasta_Turu', 'Yaş Grubu', 'segment']


# Analyze rare categories to determine if they need to be combined
rare_analyser(df_merged, "Churn", cat_cols)  # Review for rare categories

# Apply rare encoding to combine categories with a threshold of 2%
df_merged = rare_encoder(df_merged, 0.02)
rare_analyser(df_merged, "Churn", cat_cols)  # Recheck the rare categories after encoding

# One-Hot Encoding
df_merged.drop(["segment"], axis=1, inplace=True)  # Drop the 'segment' column as it was used for churn analysis, not for modeling

# Update the list of categorical columns
cat_cols = [col for col in cat_cols if col not in "segment"]
print(cat_cols)

# Apply one-hot encoding to categorical variables
df_merged = one_hot_encoder(df_merged, cat_cols, drop_first=True)
df_merged.head()  # The categorical columns have been converted into dummy variables

# Convert boolean columns to integers (0 and 1)
for col in df_merged.columns:
    if df_merged[col].dtype == 'bool':
        df_merged[col] = df_merged[col].astype(int)

# Standardize numerical columns
print(num_cols)  # ['Yaş', 'İşlem Tarihi', 'Toplam Harcama', 'Müşteri_ID']
num_cols = [col for col in num_cols if col not in ['Müşteri_ID', 'İşlem Tarihi']]  # Update the list of numerical columns

scaler = MinMaxScaler()
df_merged[num_cols] = scaler.fit_transform(df_merged[num_cols])

# The dataset is now ready for machine learning modeling

##############################################################################################################################

##################   Modeling    ###########################################################################################

# Prepare the data for modeling
y = df_merged["Churn"]
X = df_merged.drop(["Churn", "Müşteri_ID", "İşlem Tarihi", "Hizmet aldığı Branş", "Tedavi Adı"], axis=1)

# Compare base models to identify the best model for our dataset
base_models(X, y, scoring="accuracy")

"""
Accuracy: 0.7215 (LR) 
Accuracy: 0.7313 (KNN)  
Accuracy: 0.7333 (CART)  
Accuracy: 0.7586 (RF)  
Accuracy: 0.7437 (GBM) 
Accuracy: 0.7646 (XGBoost) -- 
Accuracy: 0.7573 (LightGBM)  
Accuracy: 0.7668(CatBoost) -- 

The best performing models are XGBoost and CatBoost. We will proceed with hyperparameter tuning for XGBoost and CatBoost.
"""

# Evaluate XGBoost and CatBoost before hyperparameter tuning

xgboost_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=46)
cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
print(round(cv_results["test_accuracy"].mean(), 4))
print(round(cv_results["test_f1"].mean(), 4))
print(round(cv_results["test_roc_auc"].mean(), 4))


catboost_model = CatBoostClassifier(verbose=False, random_state=46)
cv_results2 = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
print(round(cv_results2["test_accuracy"].mean(), 4))
print(round(cv_results2["test_f1"].mean(), 4))
print(round(cv_results2["test_roc_auc"].mean(), 4))

"""
XGBoost Results:

Accuracy: 0.7646
F1: 0.5165
ROC AUC: 0.7954
CatBoost Results:

Accuracy: 0.7668
F1: 0.5195
ROC AUC: 0.7977
"""

# Both models perform similarly, but we'll proceed with XGBoost due to slightly better interpretability and faster tuning.

print(xgboost_model.get_params())  # View the default hyperparameters for XGBoost

# Set up the parameter grid for hyperparameter tuning
xgboost_params = {"max_depth": [3, 6],
                  "learning_rate": [0.1, 0.01],
                  "n_estimators": [100, 500]
                  }

classifiers = [("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=46), xgboost_params)]

# Perform hyperparameter tuning
best_models = hyperparameter_optimization(X, y)

"""
After tuning:

XGBoost ROC AUC (Before): 0.79
XGBoost ROC AUC (After): 0.8
Best parameters: {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 500}
The ROC AUC score improved slightly after tuning, indicating the benefit of hyperparameter optimization.
"""

# Train the final model with the best parameters
final_model = best_models["XGBoost"]
final_model.fit(X, y)

# Predict churn for a random customer to test the model
random_user = X.sample(1, random_state=45)
final_model.predict(random_user)  # The prediction shows that the customer with ID 81326 is likely to churn

#############################################################################################################################


##################   Feature Importance    ###########################################################################################

# Feature importance analysis to identify the most impactful features in the model
plot_importance(final_model, X, num=10)
"""
Insights:
- Insurance Type (SSK) and Patient Type (Polyclinic) are the most important features in predicting churn.
- Districts like 'DİĞER' also play a significant role in churn prediction.

Further feature engineering could focus on these important features to enhance model performance."""

#######################################################################################################################


##################   Use Model for Prediction    ###########################################################################################

# Save the final model for future use
joblib.dump(final_model, "final_model.pkl")

# Load the saved model for use in future predictions
new_model = joblib.load("final_model.pkl")

# Test the model with a new random customer sample
random_user = X.sample(1, random_state=10)  # Customer ID 236918
new_model.predict(random_user)  # The prediction shows that the customer with ID 236918 is not likely to churn

# Verify the churn status of the customer with ID 236918
new_id = 236918
ischurn = df_merged[df_merged["Müşteri_ID"] == new_id]["Churn"]
print(f"Müşteri ID {new_id} için Churn durumu: {ischurn.iloc[0]}")

# The customer with ID 236918 is confirmed not to churn based on the model prediction.

#########################################################################################################################