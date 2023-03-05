# Imports

from fastapi import FastAPI
import pickle, uvicorn, os
import uvicorn
from pydantic import BaseModel
# from sklearn.preprocessing import OneHotEncoder as ohe
import pandas as pd
import numpy as np

#########################################################################
# Config & setup


## Variable of environment (path of ml items)
DIRPATH = os.path.dirname(__file__)
ASSETSDIRPATH = os.path.join(DIRPATH, 'Assets')
ML_ITEMS_PKL = os.path.join(ASSETSDIRPATH, 'ML_items.pkl')

print(f"{'*'*10} Config {'*'*10}\n INFO: DIRPAHT = {DIRPATH} \n INFO: ASSETSDIRPATH = {ASSETSDIRPATH} ")


## API Basic config
app = FastAPI(title = 'CUSTOMER CHURN API',
              version = '0.0.1',
              description = 'Predicting customer churn')


## Loading of assets
with open(ML_ITEMS_PKL,"rb") as f:
    loaded_items = pickle.load(f)
print("INFO: Loaded assets:", loaded_items)

model = loaded_items['model']
scaler = loaded_items['scaler']
numeric_columns = loaded_items['numeric_columns']

#########################################################################
# API core


## BaseModel
class ModelInput(BaseModel):
    # List of columns and data types
    MONTANT : float
    FREQUENCE_RECH : float
    REVENUE : float
    ARPU_SEGMENT : float
    FREQUENCE : float
    DATA_VOLUME : float
    ON_NET : float
    ORANGE : float
    TIGO : float
    REGULARITY : int
    FREQ_TOP_PACK : float
    TENURE2 : int
    
## Utilities

def feature_engeneering(
    dataset, scaler, 
):  # FE : ColumnTransfromer, Pipeline
    "Cleaning, Processing and Feature Engineering of the input dataset."
    """:dataset pandas.DataFrame"""

    output_dataset = dataset.copy()

    output_dataset = scaler.transform(output_dataset)

    return output_dataset


def make_predict(MONTANT,
                 FREQUENCE_RECH,
                 REVENUE,
                 ARPU_SEGMENT,
                 FREQUENCE,
                 DATA_VOLUME,
                 ON_NET,
                 ORANGE,
                 TIGO,
                 REGULARITY,
                 FREQ_TOP_PACK,
                 TENURE2):
    ""
    df = pd.DataFrame([[MONTANT,
                        FREQUENCE_RECH,
                        REVENUE,
                        ARPU_SEGMENT,
                        FREQUENCE,
                        DATA_VOLUME,
                        ON_NET,
                        ORANGE,
                        TIGO,
                        REGULARITY,
                        FREQ_TOP_PACK,
                        TENURE2]], 
                        
                        columns = numeric_columns)
                        
    
    X = feature_engeneering(dataset=df, scaler=scaler,)
    model_output = model.predict(X).tolist()
    return model_output

    # X = df
    # output = model.predict(X).tolist()
    # return output
   
    
## Endpoints
@app.post("/CUSTOMER CHURN")
async def predict(input:ModelInput):
    """ __descr__

    __datails__
    """
    output_pred = make_predict(MONTANT = input.MONTANT,
                        FREQUENCE_RECH = input.FREQUENCE_RECH,
                        REVENUE = input.REVENUE,
                        ARPU_SEGMENT = input.ARPU_SEGMENT,
                        FREQUENCE = input.FREQUENCE,
                        DATA_VOLUME = input.DATA_VOLUME,
                        ON_NET = input.ON_NET,
                        ORANGE = input.ORANGE,
                        TIGO = input.TIGO,
                        REGULARITY = input.REGULARITY,
                        FREQ_TOP_PACK = input.FREQ_TOP_PACK,
                        TENURE2 = input.TENURE2,
                          )
    
    if (output_pred[0]>0):
        output_pred ="This customer is not loyal he/she will leave your company."
    else:
        output_pred ="This is a LOYAL customer he/she will stay."
    return {
        "prediction": output_pred,
       # "input": input,
    }
    


#########################################################################
# Execution

if __name__=='__main__':
    uvicorn.run('api:app',
            reload = True,
            )