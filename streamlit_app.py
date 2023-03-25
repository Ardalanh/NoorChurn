import streamlit as st
from PIL import Image
import pandas as pd
import pathlib
import requests
import joblib
import os

st.set_page_config(
    "Churn Prediction by Ardalan",
    "📊",
    initial_sidebar_state="expanded",
    layout="wide",
)

clf_path = "clf.joblib"

class CategoryTransformer:
    def __init__(self, maps, col_name):
        self.category_maps = maps
        self.col_name = col_name
        
    def transform(self, X, **transform_params):
        for key, val in self.category_maps.items():
            X[self.col_name].replace(key, val, inplace=True)
        return X
    
    def fit(self, X, y= None, **fit_params):
        return self
    
    def fit_transform(self, X, y= None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

@st.cache_resource 
def load_model():
    return joblib.load(clf_path)

@st.cache_data 
def load_sample_data():
    return pd.read_csv("sample_data.csv", index_col="CustomerID")

model = load_model()


def main() -> None:
    download_dependencies()

    tab1, tab2, tab3 = st.tabs(["EDA", "Modeling", "API endpoint(GUI)"])
    with tab1:
        st.header("Overview")
        st.caption("First I started to explore the data and target variable Churn.")
        st.caption("Reading the column descriptions gave me an overall understandng of the data, altough I lack the domain knolwedge to dive really deep.")
        st.caption("First important thing was that the target variable Churn is unbalanced")
        with st.expander("Graph"):
            st.image("churn.png")
        st.caption("This information is useful for me to make decision on choice of the model.")

        st.header("Numeric Columns")
        st.caption("I decided to look at the distribution of numeric columns")
        with st.expander("Graph"):
            st.image(Image.open("NumericDist.png"))
        st.caption("I can see that there are outliers in Tenure, Warehousetohome, NumberOfAddress, CouponUsed, DaysSinceLastOrder and CashbackAmount.")
        st.caption("This is relevant to our choice of model or wether we want to remove these outliers or no.")
        st.caption("Now I want to understand what's the distribution difference between Churn 1 nd 0")
        with st.expander("Graph"):
            st.image(Image.open("NumericDistChurn.png"))
        st.caption("Here we can see that there is noticable difference between people who churned and who didn't.")
        st.caption("Tenure is our first candidate for the best predictor of churn.")
        st.caption("The second best predictor seems to the Complain column.")
        st.caption("Following those columns, SatisfactionScore and CityTier seem to show some information about churn.")

        st.header("Categoric Columns")
        st.caption("We see here that few categories from PreferredOrderCat, Marital Status and PreferredPaymentMonde columns are correlated with Churn")
        st.caption("Also I'm noticing few duplicated categories in PreferredLoginDevice and PrefredOrderCat")
        with st.expander("Graph"):
            st.image(Image.open("CategoricDist.png"))

        st.header("Missing Values")
        st.caption("initially it seems that the missing values are completely at random but after closely analyzing the missing values I figured that if I sorted the data by CashbackAMount, we can see it's not completely at Random")
        st.caption("I tried to understand if these missing values are random or not, and since I don't have the context where and how this data was collected and what each column is exactly referring to I can't make better judgement on the type of missing data, I'm going to treat this as missing at random.")
        st.caption("Although missing in the churn column is completely at random")
        with st.expander("Graph"):
            st.image(Image.open("Missing.png"))

        st.header("Duplicated rows")
        st.caption("The data includes around +400 duplicated rows, if all duplicated rows are in train there will be no issue but randomly splitting into train/test there will be data leakage.")
        st.caption("Data leakage from train to test will give us unreliable and elevated test metrics.")

    with tab2:
        st.header("Preprocessing")
        st.caption("I fixed the columns PreferredLoginDevice and PreferedOrderCat, where there were duplicated values Phone, Mobile Phone and mobile.")
        st.caption("I filled the na of the numeric columns with median values and na of the categoric columns with most frequent values.")
        st.caption("I onehot encoded the categoric columns and left the numeric columns as is.")

        st.header("Modeling")
        st.caption("I have decided to use tree based models because we want to have simple enough algorithm that can have good predictive power and work well on unbalanced data.")
        st.caption("The data does not seem complicated and number of rows is not high, RandomForest is more than enough to get at least a good benchmark.")
        st.caption("As we shall see that the accuracy of the model is quite high, if we were not getting high accuracy that would be our signal to do more data pre-processing, feature engineering and use other models.")
        st.caption("I made the preprocessing pipeline for training and prediction time.")
        st.caption("I also search through hyperparaters of RandomForest and evaluated the model in 5 cross validation folds.")
        st.caption("For the metric of evaluation I used F1Score because it's suitable for unbalanced data and when we care about equally percision and recall of all casses.")
        with st.expander("Result Report"):
            st.image(Image.open("Results.png"))
        st.caption("The results of 0.95 of F1 score is really good, and it seems to me it's too good.")
        st.caption("I believe either this data is synthetic or there is a data leakage that I haven't noticed.")
        st.caption("Usually in reallity models with this level of accuracy are quite rare")
        st.caption("One of the positive side of using RandomForest as our model is that we can sort our features by their importance for prediction of Churn.")
        with st.expander("Feature Importance"):
            st.image(Image.open("FeatureImportance.png"))
        st.caption("Here we can see how much each column is affecting the probability of Churning.")

    with tab3:
        uploaded_file = st.file_uploader("Choose a file", "csv")
        if uploaded_file:
            X = pd.read_csv(uploaded_file, index_col="CustomerID")
            pred = model.predict_proba(X)[:,1]
            pred = pd.DataFrame(pred, columns=["Probability of Churn"], index=X.index)
            st.write(pred)
            st.download_button(
                    label="Download results as CSV",
                    data=pred.to_csv().encode("utf-8"),
                    file_name='results.csv',
                    mime='text/csv',
                )

        st.download_button(
                label="Download sample CSV",
                data=load_sample_data().to_csv().encode("utf-8"),
                file_name='sample_data.csv',
                mime='text/csv',
            )

def download_dependencies():
    images = ["https://raw.githubusercontent.com/Ardalanh/NoorChurn/main/images/CategoricDist.png",
              "https://raw.githubusercontent.com/Ardalanh/NoorChurn/main/images/FeatureImportance.png",
              "https://raw.githubusercontent.com/Ardalanh/NoorChurn/main/images/Missing.png",
              "https://raw.githubusercontent.com/Ardalanh/NoorChurn/main/images/NumericDist.png",
              "https://raw.githubusercontent.com/Ardalanh/NoorChurn/main/images/NumericDistChurn.png",
              "https://raw.githubusercontent.com/Ardalanh/NoorChurn/main/images/Results.png",
              "https://raw.githubusercontent.com/Ardalanh/NoorChurn/main/images/churn.png"]
    for img in images:
        file_path = os.path.basename(img)
        if os.path.exists(file_path):
            continue
        __download_url(img, file_path)

    weight = "https://raw.githubusercontent.com/Ardalanh/NoorChurn/main/clf.joblib"
    file_path = os.path.basename(weight)
    if not os.path.exists(file_path):
        __download_url(weight, file_path)

    sample_data = "https://raw.githubusercontent.com/Ardalanh/NoorChurn/main/sample_data.csv"
    file_path = os.path.basename(sample_data)
    if not os.path.exists(file_path):
        __download_url(sample_data, file_path)


def __download_url(url, path):
    img_data = requests.get(url).content
    with open(path, "bw+") as f:
        st.write(path)

if __name__ == "__main__":

    main()