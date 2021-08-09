import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import numpy as np
import streamlit as st
import pandas as pd
import pyarrow as pa
from numpy import linalg as LA
import base64


def missing_values(dataframe):
    """
    Creates a dataframe of the columns that contain missing values (nan) and their correspoding percentage of values missing
    params: dataframe
    returns: dataframe with 2 columns (Missing values, % of total values that are missing)
    """
    mis_val = dataframe.isnull().sum()
    mis_val_percent = 100 * dataframe.isnull().sum() / len(dataframe)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    st.write("Your selected dataframe has " + str(dataframe.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
        " columns that have missing values.")
    return mis_val_table_ren_columns


def heat_map(dataframe):
    """
    Plots the heatmap of the dataframe
    params: dataframe
    returns: Matplotlib.pyplot Figure
    """
    fig, ax = plt.subplots()
    correlation = dataframe.corr()
    ax = sns.heatmap(correlation, cmap="coolwarm")
    return fig


def plot_histogram(dataframe, feature):
    """
    Plots the histogram of a given feature
    params: dataframe, specified feature
    returns: Matplotlib.pyplot Figure
    """
    fig, ax = plt.subplots()
    ax = sns.histplot(data=dataframe, x=feature, kde=True)
    return fig


def histogram(dataframe):
    """
    Plots histogram(s) of specified features in a dataframe. If multiple features are selected, then 
    output pairwise histograms. Warns user of attempting to pairwise plot perfectly correlated features  
    param: dataframe
    returns: none
    """
    plot_features = st.multiselect(label="Select the features to visualize", options=dataframe.columns)

    try:                               
        if len(plot_features) == 1:
            st.write(plot_histogram(dataframe, plot_features[0]))
        
        if len(plot_features) > 1:
            pairwise = dataframe[plot_features].copy()
            fig = sns.PairGrid(data=pairwise)
            fig.map_upper(sns.kdeplot, fill=True)
            fig.map_lower(sns.kdeplot, fill=True)
            fig.map_diag(sns.histplot, kde=True)
            st.pyplot(fig)
    except LA.LinAlgError as e:
        st.write("Singular Matrix error due to perfect correlation, refer to heat map and adjust accordingly")


def label(dataframe, col):
    """
    Applies label encoding for the specified column in a dataframe
    params: dataframe, specified column
    returns: dataframe with label encoding on the specified column
    """
    dataframe[col] = LabelEncoder().fit_transform(dataframe[col])
    return dataframe


def one_hot(dataframe, col):
    """
    Applies one hot encoding for the specified column in a dataframe
    params: dataframe, specified column
    returns: dataframe with one-hot encoding on the specified column
    """
    return pd.get_dummies(dataframe, columns=[col])


def describe(dataframe):
    """
    Outputs the body text for the original data section in the web application. Prompts user for 
    label encoding or one-hot encoding if there is categorical data (otherwise summary statistics would
    not be possible)
    params: dataframe
    returns: same or altered (encoded) dataframe
    """
    st.write("Shape of data:", dataframe.shape)
    st.write("Peek at the original data\n", dataframe.head())
    try:
        st.write("Summary Statistics\n", dataframe.describe())
    except pa.lib.ArrowInvalid as e:
        st.write("First convert any categorical features")
        categorical_cols = list(set(dataframe.columns) - set(dataframe._get_numeric_data().columns))
        for col in categorical_cols:
            strategy = st.radio(label=f'Select encoding strategy for the column {col}', options=("Label Encoding", "One-Hot Encoding"))
            if strategy == "Label Encoding":
                dataframe = label(dataframe, col)
            else:
                dataframe = one_hot(dataframe, col)
        st.write("Encoded data\n", dataframe.head())
        st.write(dataframe.describe())
    st.write(missing_values(dataframe))
    st.write("Heat map of features")
    st.pyplot(heat_map(dataframe))
    st.write("Histogram of features")
    histogram(dataframe)
    return dataframe


def mutate_describe(dataframe, y):
    """
    Outputs the body text for the mutated data section in the web application
    params: dataframe, y (target data)
    returns none
    """
    st.write("Shape of new data:", dataframe.shape)
    st.write("Shape of target:", y.shape)
    st.write("Peek at the new data\n", dataframe.head())
    st.write("Peek at the target\n", y.head())
    st.write("Summary Statistics\n", dataframe.describe())
    st.write(missing_values(dataframe))
    st.write("Heat map of new features\n", heat_map(dataframe))

    
def drop(dataframe):
    """
    Drops the selected columns from the dataframe
    params: dataframe
    returns: same of altered dataframe
    """
    features = st.multiselect(label="Select features to drop", options=dataframe.columns)                
    for feature in features:
        del dataframe[feature]
    return dataframe


def imputer(dataframe):
    """
    Applies SimpleImputer on each column that has nan values. Applies user desired strategy
    params: dataframe
    returns: same or imputed dataframe
    """
    nan_cols = dataframe.columns[dataframe.isna().any()].tolist()
    if len(nan_cols) > 0:
        strategy = st.radio(label="Select strategy (note PCA will fail if there are missing values in data)", options=("None", "Mean", "Median", "Most Frequent"))
        if strategy == "None":
            return dataframe
        elif strategy == "Mean":
            return pd.DataFrame(SimpleImputer().fit_transform(dataframe), columns=dataframe.columns)
        elif strategy == "Median":
            return pd.DataFrame(SimpleImputer(strategy="median").fit_transform(dataframe), columns=dataframe.columns)
        else:
            return pd.DataFrame(SimpleImputer(strategy="most_frequent").fit_transform(dataframe), columns=dataframe.columns)
    st.write("No columns with missing values, so Imputer does not need to be applied")
    return dataframe


def pca(dataframe):
    """
    Applies PCA to dataframe, if user wishes to do so. Dimensions decided
    by how much variance the user wishes to preserve.
    params: dataframe 
    returns: same or dimensionality reduced dataframe
    """
    enable = st.checkbox(label="Enable Principal Component Analysis (PCA)")
    if enable:
        variance = st.slider(label="Select the amount of variance you wish to preserve", min_value=0.01, max_value=0.99, step=0.01)
        return pd.DataFrame(PCA(n_components=variance).fit_transform(dataframe))
    return dataframe


def standardize(dataframe):
    """
    Standardizes dataframe, if user wishes to do so
    params: dataframe 
    returns: same or standardized dataframe
    """
    enable = st.checkbox(label="Enable to apply standardization")
    if enable:
        return pd.DataFrame(StandardScaler().fit_transform(dataframe))
    return dataframe


def normalize(dataframe):
    """
    Normalizes dataframe, if user wishes to do so
    params: dataframe 
    returns: same or normalized dataframe
    """
    enable = st.checkbox(label="Enable to apply normalization")
    if enable:
        return pd.DataFrame(MinMaxScaler().fit_transform(dataframe))
    return dataframe


def get_table_download_link(df, target):
    """
    Generates a link allowing the data in a given panda dataframe to be downloaded
    params:  dataframe
    returns: tuple of href strings (X, y)
    """
    X_csv = df.to_csv()
    X_b64 = base64.b64encode(X_csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    X_href = f'<a href="data:file/csv;base64,{X_b64}" download="X_data.csv">Download feature data (csv)</a>'

    y_csv = target.to_csv()
    y_b64 = base64.b64encode(y_csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    y_href = f'<a href="data:file/csv;base64,{y_b64}" download="y_data.csv">Download target data (csv)</a>'
    return X_href, y_href


def mutate(dataframe):
    """
    Performs the mutation of the dataframe
    params: dataframe
    returns: none
    """
    st.write("Target features")
    target = st.multiselect(label="Select target features", options=dataframe.columns)
    y = dataframe[target]
    dataframe = dataframe.drop(target, axis=1)
    
    st.write("Drop feature(s)")
    drop(dataframe)
    
    st.write("Imputation")
    dataframe = imputer(dataframe)


    st.write("Principle Component Analysis")
    dataframe = pca(dataframe) 

    columns = dataframe.columns

    st.write("Standardize")
    dataframe = standardize(dataframe)
    
    st.write("Normalize")
    dataframe = normalize(dataframe)
        
    dataframe.columns = columns
    mutate_describe(dataframe, y)
    if len(target) >= 1:
        st.write("Histogram of new features")
        histogram(dataframe)
    else:
        st.write("To view new histograms, please select target and/or alter the data. Otherwise simply refer to histograms from eariler")

    st.write("Download adjusted data and target")
    X_link, y_link = get_table_download_link(dataframe, y)
    st.markdown(X_link, unsafe_allow_html=True)
    st.markdown(y_link, unsafe_allow_html=True)