import streamlit as st
import pandas as pd
from output import *

st.title("Data Visualization")

st.write("""
Welcome to my Data Visualization web application. Inspecting data can be tedious and difficult,
so why not streamline the whole process? Just drop any tabular data file (csv) and play around with it.
It's that simple.
""")

uploaded_file = st.file_uploader(label="Upload csv file", type="csv", help="Make sure you are uploading only a single csv file. To test a new csv file simply upload another file.")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.header("Explore the Data")
    dataframe = describe(dataframe)

    st.markdown('#')
    should_mutate = st.checkbox(label="Would you like to mutate the data?")
    if should_mutate:
        st.header("Mutate Data")
        mutate(dataframe)

