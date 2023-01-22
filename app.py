import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer, VectorAssembler, Normalizer, StandardScaler, MinMaxScaler
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StringType, DateType
from pyspark.sql.functions import to_date, datediff
from pyspark.sql.functions import concat, lit, avg, split, isnan, when, count, col, sum, mean, stddev, min, max, round
from pyspark.sql import Window
from pyspark.ml.classification import LogisticRegression, GBTClassifier, NaiveBayes, RandomForestClassifier, LinearSVC, DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import Bucketizer
from pyspark.ml.classification import RandomForestClassificationModel
import streamlit as st
import csv

spark = SparkSession.builder.appName('customer_retention') \
            .getOrCreate()
sqlContext = SQLContext(spark)


st.set_page_config(layout="wide")
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title('''Customer Retention Analysis for Music Streaming Services''')
st.subheader( 'Github repo [here](https://github.com/Sapphirine/202212-29-Customer-Retention-Analysis-for-Music-Streaming-Services)')

#Take csv file as input
uploaded_file = st.file_uploader("Choose a file", type="csv")

#function to create features from input data
def create_features(uploaded_file):
    stringIndexerGender = StringIndexer(inputCol="gender", outputCol="genderIndex", handleInvalid = 'skip')
    stringIndexerLevel = StringIndexer(inputCol="last_level", outputCol="levelIndex", handleInvalid = 'skip')
    stringIndexerState = StringIndexer(inputCol="last_state", outputCol="stateIndex", handleInvalid = 'skip')
    CategoricalFeatures = ['gender', 'last_level', 'last_state']
    indexers = [StringIndexer(inputCol = col,
    outputCol = "{}Index".format(col))\
                          for col in CategoricalFeatures]
    encoder = [OneHotEncoder(inputCols=["genderIndex", "last_levelIndex", "last_stateIndex"],
                                       outputCols=["genderVec", "levelVec", "stateVec"],
                                handleInvalid = 'keep')]

    features = ['genderVec', 'levelVec', 'stateVec', 'days_active', 'avg_songs', 'avg_events', 'thumbs_up', 'thumbs_down', 'addfriend']
    assembler = [VectorAssembler(inputCols=features, outputCol="rawFeatures")]

    normalizer = Normalizer(inputCol="rawFeatures", outputCol="features", p=1.0)

    preprocessor = Pipeline(stages = indexers + encoder + assembler + [normalizer]).fit(uploaded_file)
    preprocessed_df = preprocessor.transform(uploaded_file)
    preprocessed_df = preprocessed_df.withColumnRenamed("last_levelIndex", "levelIndex")\
       .withColumnRenamed("last_stateIndex", "stateIndex")
    
    return preprocessed_df
    
#function to load model and predict
def trained_model(test):
                rf_model = RandomForestClassificationModel.load("model")
                rf_pred_test = rf_model.transform(test).cache()
                preds = rf_pred_test.select("prediction")
                if preds == 1:
                        results = 1
                else:
                        results = 0
                return results

#If file uploaded then 
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    data = create_features(sqlContext.createDataFrame(dataframe))
    st.text('Our target variable is churn and we are giving vectorized data to the model.')
    if st.button('Predict', key='1'):
                data = data.withColumnRenamed("churn", "label")
                results_data = trained_model(data)
                if results_data == 1:
                        st.write("The user is likely to churn")
                else:
                        st.write("The user is not likely to churn")

#Alternate option to take user input manually                       
st.write("OR")
st.write("Enter Attributes")
uid = st.number_input("User Id")
gender = st.radio("Gender", ('M', 'F'))
level = st.radio("Subscription Level", ('Free','Paid'))
active_days = st.number_input("Active days")
state = st.selectbox(
    'Last State',
    ('PA','TX', 'FL', 'WI', 'IL', 'NC', 'SC', 'AZ', 'CT', 'NH', 'OTHER'))
avg_songs = st.number_input("Avg Songs")
avg_events = st.number_input("Avg Events")
thumbsup = st.number_input("Thumbs Up")
thumbsdown = st.number_input("Thumbs Down")
add_friend = st.number_input("Add Friend")
fields = [uid, gender, level,active_days, state, avg_songs, avg_events, thumbsup, thumbsdown, add_friend]
 
#actions to follow when predict button is clicked
if st.button('Predict', key='2'):
            with open('user.csv','a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(fields)
            df = spark.read.format("csv").options(header="false", inferschema="true").load("user.csv")
            d = {
                        "userId": [uid],
                        "gender": [gender],
                        "churn": "0",	
                        "last_level": [level],	
                        "days_active": [active_days],	
                        "last_state": [state],	
                        "avg_songs": [avg_songs],	
                        "avg_events": [avg_events],	
                        "thumbs_up": [thumbsup],	
                        "thumbs_down": [thumbsdown],	
                        "addfriend": [add_friend]
            }
            df = pd.DataFrame(data=d)
            data = create_features(sqlContext.createDataFrame(df))
            data_ml = data.withColumnRenamed("churn", "label")
            results_data = trained_model(data_ml)
            if results_data == 1:
                        st.write("The user is likely to churn")
            else:
                        st.write("The user is not likely to churn")
 
            
