import streamlit as st
import os, pickle
import pandas as pd


# first line after the importation section
st.set_page_config(page_title="CUSTOMER CHURN WEB APP", page_icon="🌼", layout="centered")
DIRPATH = os.path.dirname(os.path.realpath(__file__))

# Useful functions
# @st.cache()  # stop the hot-reload to the function just below
# def setup(tmp_df_file):
#     "Setup the required elements like files, models, global variables, etc"
#     # sepal_length (cm)	sepal_width (cm)	petal_length (cm)	petal_width (cm)
#     pd.DataFrame(
#         dict(
#             sepal_length=[],
#             sepal_width=[],
#             petal_length=[],
#             petal_width=[],
#         )
#     ).to_csv(tmp_df_file, index=False)

    

@st.cache_data(allow_output_mutation=True)
def load_ml_items():
    "Load ML items to reuse them"

    with open(os.path.join('Assets','ML_items'), 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object



#Config 
# tmp_df_file = os.path.join(DIRPATH, '..' ,"tmp", "data.csv")
# setup(tmp_df_file)

loaded_object = load_ml_items()
if 'results' not in st.session_state:
    st.session_state['results'] = [] # This is to store all the predictions


# Interface and logic
st.title("Customer churn")

st.sidebar.write(f"Customer churn App")
st.sidebar.write(f"This classification app has been made with Streamlit.")

form = st.form(key="information", clear_on_submit=True) # After every prediction, the valuew will reset

# Form to retreive the inputs
with form:
    # First row
    cols = st.columns((1, 1))
    MONTANT = cols[0].number_input('MONTANT')
    FREQUENCE_RECH = cols[1].number_input('FREQUENCE_RECH')

    # Second row
    cols = st.columns((1, 1))
    REVENUE = cols[0].number_input('REVENUE')
    ARPU_SEGMENT = cols[1].number_input('ARPU_SEGMENT')

    # Third row
    cols = st.columns((1,1))
    FREQUENCE = cols[0].number_input('FREQUENCE')
    DATA_VOLUME = cols[1].number_input('DATA_VOLUME')

    # Forth row
    cols = st.columns((1,1))
    ON_NET = cols[0].number_input('ON_NET')
    ORANGE = cols[1].number_input('ORANGE')

    # Fith row
    cols = st.columns((1,1))
    TIGO = cols[0].number_input('TIGO')
    REGULARITY = cols[1].number_input('REGULARITY')

    # Sixth row
    cols = st.columns((1,1))
    FREQ_TOP_PACK = cols[0].number_input('FREQ_TOP_PACK')
    TENURE2 = cols[1].number_input('TENURE2')
    
    #Submit button
    submitted = st.form_submit_button(label="Predict")

# Logic when the inputs are submitted
if submitted:

    # Inputs formatting
    dict_input = {
        'MONTANT': [MONTANT],
        'FREQUENCE_RECH': [FREQUENCE_RECH],
        'REVENUE': [REVENUE],
        'ARPU_SEGMENT': [ARPU_SEGMENT],
        'FREQUENCE':[FREQUENCE],
        'DATA_VOLUME':[DATA_VOLUME],
        'ON_NET':[ON_NET],
        'ORANGE':[ORANGE],
        'TIGO':[TIGO],
        'REGULARITY':[REGULARITY],
        'FREQ_TOP_PACK':[FREQ_TOP_PACK],
        'TENURE2':[TENURE2]
        }
        
    df = pd.DataFrame.from_dict(dict_input)
    
     # Preprocessing
    scaled_df = loaded_object['scaler'].transform(df)
     
     # Prediction
    output = loaded_object['model'].predict_proba(scaled_df)
     
     # print(type(output))
     # print(output)
     # print(output.shape)
     
     # Format the prediction output
    st.write('DataFrame: Input and prediction details')
    pred_class = output.argmax(axis=-1)
    confidence_score = output[0, pred_class] # np.array[axis_0, axis_1
     
    df["confidence score"] = confidence_score # [0.8]
    df["predicted class"] = pred_class # [0]
    st.session_state['results'].append(df)
     
     # Add pred label later

    
    # Display prediction results
    st.balloons()
    st.success(f"Predicted class: {pred_class[0]}")
    st.success(f"Confidence score: {confidence_score[0]}")

    # Area to visualize the previous predictions 
    expander = st.expander("See the predictions done until now..")
    with expander:
        result = pd.concat(st.session_state['results'],)
        st.dataframe(result, use_container_width=True)