# TELECOMMUNICATION CUSTOMER CHURN CLASSIFICATION (MACHINE LEARNING)
# OBJECTIVE
This challenge is for an African telecommunications company that provides customers with airtime and mobile data bundles. The objective is to develop a machine learning model to predict the likelihood of each customer “churning,” i.e. becoming inactive and not making any transactions for 90 days. This solution will help this telecom company to better serve their customers by understanding which customers are at risk of leaving.
The dataset Variable Definition
The churn dataset includes 19 variables including 15 numeric variables and 04 categorical variables.
user_id – User ID
REGION - The location of each client
TENURE - Duration in the network
MONTANT - Top-up amount
FREQUENCE_RECH - Number of times the customer refilled.
REVENUE - Monthly income of each client
ARPU_SEGMENT - Income over 90 days / 3
FREQUENCE - Number of times the client has made an income.
DATA_VOLUME - Number of connections
ON_NET - Inter expresso call
ORANGE - Call to orange
TIGO - Call to Tigo
ZONE1 - Call to zones1
ZONE2 - Call to zones2
MRG - A client who is going.
REGULARITY - Number of times the client is active for 90 days.
TOP_PACK - The most active packs
FREQ_TOP_PACK - Number of times the client has activated the top pack packages
CHURN – Target variable to predict.


# DATA PROCESSING
EXPLORATORY DATA ANLYSIS 
Some of the columns had most of the data missing. These columns include ZONE1, ZONE2, user_id, TOP_PACK, MRG. These columns had at least more than 50% of the data missing. For this classification project, the user_id column will not be needed so it must also be dropped or deleted. To drop the columns you use this code; df.drop(['ZONE1', 'ZONE2','user_id', 'TOP_PACK','MRG'], axis = 1, inplace = True)

Most of the data was this dataset were missing or not provided. Most of the missing data were in the numeric columns. Such columns include. FREQUENCE, ON_NET, ARPU_SEGMENT and others. In filling out these null values the mean and median of that column was used. For example to fill the null values in the FREQUENCE  column with the mean, use df['FREQUENCE'] = df['FREQUENCE'].fillna((df['FREQUENCE'].mean())). 

# SPLITTING THE DATA SET AND FEATURE ENGINEETING
The data was split into train and evaluation sets in the proportion of 80% and 20% respectively. 
Since all remaining columns are numerical there will not be the need to encode them. But before training, the numeric columns have to be scaled. Feature scaling in machine learning is one of the most critical steps during the pre-processing of data before creating a machine learning model. Scaling can make a difference between a weak machine learning model and a better one. The most common techniques of feature scaling are Normalization and Standardization. In our case we’ll use the Standard scaler.

# MODEL TRAINING
Several classification models were used for the training. Most of them performed very well. The best model used used random forest.


From all the models that was trained, the random forest model, compared to the other models had the best score in terms of accuracy. It had a prediction accuracy of 86% (ahead of the others in decimals). Naive Bayes didn’t have a good accuracy score but had a very good recall score of 96%. All other models had similar scores that had accuracy scores within 85 - 86%, recall score within 50 - 85%, Precision score within 45 - 60%, F1 score within 50 - 65% and F2 score within 50 - 70%.


# API FOR CUSTOMER CHURN MACHINE LEARNING MODEL

# Setup
Please follow the instructions to setup this project on your local machine.

You need to have Python 3 on your system (a Python version lower than 3.10). Then you can clone this repo and being at the repo's root :: repository_name> ... follow the steps below:

Windows:

    python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  

Linux & MacOs:

    python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  
Execution
To run the project, please execute one of the following commands having the venv activated.

# FastAPI

python's command

    python src/main.py 

uvicorn's command

    uvicorn scr.main:app 

## Accessing the app in your browser
[Click here to open in your browser](http://127.0.0.1:8000/)

[Open the api's documentation in the browser](http://127.0.0.1:8000/docs)




# STREAMLIT WEB APP FOR CUSTOMER CHURN CLASSIFICATION MODEL

This is a streamlit app developed to create an interface for a machine learning model used to predict customer churn in a telecommunication company.

# Setup
To setup this project, you need to have Python3 on your system. Then you can clone this repo and being at the repo's root :: repo_name> ... follow the steps below:

Windows:

    python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  
Linux & MacOs:

    python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  
NB: For MacOs users, please install Xcode if you have an issue.

## Execution
Run the Streamlit app (being at the repository root):

    streamlit run src/streamlit_app.py
Go to your browser at the following address :
http://localhost:8501

## Screenshoots
Here are some screenshoots of the app interface

![](/screenshoots/streamlit1.png)

![](/screenshoots/streamlit2.png)

Author
- [Bernard Ampomah]()