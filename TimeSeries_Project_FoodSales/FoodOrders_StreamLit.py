import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# main_bg = "Global_network_generated.jpg"
# main_bg_ext = "jpg"
# st.markdown(
#     f"""
#     <style>
#     .reportview-container {{
#         background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
#     }}
#    .sidebar .sidebar-content {{
#         background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
# )
sns.set_style("whitegrid")

col1, col2= st.columns(2)

with col1:
    st.write("""
    # Awesome Food Demand Forecasting APP!

    Powered by **Sklearn, StatsModels**
    """
    )
with col2:
    st.image('network.jpg', width=400)



st.write("---")


st.sidebar.header("User input")
uploaded_file = st.sidebar.file_uploader("Upload your file here")

df = pd.read_csv('time_series.csv', index_col=0)
df_meal = pd.read_csv('meal_info.csv')
meal_ids = list(df['meal_id'].unique())
meal_id = meal_ids[0]
forecasting_days = 20
model_list=[RandomForestRegressor,DecisionTreeRegressor,  GradientBoostingRegressor, XGBRegressor]
model_selector = ['RandomForest', 'DecisionTree', 'GradientBoosting', 'XGBoost']
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, index_col=0)
else:
    def user_input_features():

        meal_id = st.sidebar.selectbox('Meal ID',[*meal_ids])
        model_type = st.sidebar.selectbox('Models', model_selector)
        forecasting_days = st.sidebar.slider('Forecasting_weeks', min_value=0, max_value=20, value=20)
        jump = st.sidebar.slider('Train Faster', min_value=1, max_value=20, value=1)
        # training_days = st.sidebar.slider('Training_days', min_value=0, max_value=145, value=124)

        data = {'meal_id': meal_id, 'model_type':model_type, 'jump':jump, 'forecasting_weeks':forecasting_days}
        df_features = pd.DataFrame(data, index=[0])
        return df_features
    input_df = user_input_features()
    meal_id = input_df['meal_id'][0]
    jump = input_df['jump'][0]
    model_id = model_selector.index(input_df['model_type'][0])
    model = model_list[model_id]
    # training_days = input_df['training_days'][0]
    forecasting_days = input_df['forecasting_weeks'][0]
meal = ' '.join(df_meal[df_meal['meal_id']==meal_id][['category', 'cuisine']].values[0])
st.write("Displaying for meal:", meal_id, meal)
df = df.sort_values(['meal_id', 'week'])
df = df[df['meal_id']==meal_id]




max_week_quarter = 12  # maximum number of weeks in a quarter
max_week_month = 4  # maximum number of weeks in a month
df = pd.concat([pd.Series(np.sin(2 * np.pi * df['week'] / max_week_quarter), name ='week_sin_quarter'), df], axis =1)
df = pd.concat([pd.Series(np.cos(2 * np.pi * df['week'] / max_week_quarter), name ='week_cos_quarter'), df], axis =1)
df = pd.concat([pd.Series(np.sin(2 * np.pi * df['week'] / max_week_month), name ='week_sin_month'), df], axis =1)
df = pd.concat([pd.Series(np.cos(2 * np.pi * df['week'] / max_week_month), name ='week_cos_month'), df], axis =1)
fig= plt.figure(figsize=(12,4),dpi=150)
ax=sns.lineplot(data=df, x='week', y='num_orders', marker='.')
ax.set_xlim((0, 175))
# st.pyplot(fig)


#prepare time series for training
def split_sequences(sequences, context_len, pred_len, jump, pred_column):
    X, y = [], []
    for i in range(0, len(sequences), jump):
        # find pred window (start, end). 
        start = i + context_len
        end = start + pred_len
        # check if we are beyond the dataset
        if end > len(sequences):
            break
        # seq_x: i~start, seq_y: start~end
        seq_x, seq_y = sequences.iloc[i:start], sequences.iloc[start:end][pred_column]
        # seq_x = seq_x[seq_x.columns.difference([pred_column])]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# st.write("Modeling...")
st.write("---")
context_len, pred_len = 20, 20
jump = jump
X, y = split_sequences(df, context_len, pred_len, jump, 'num_orders')
X = X.reshape(X.shape[0], -1) #Flatten time and dim for ML models. 107, 200

# training_times = int(training_days//jump)
# st.write(len(df), training_days, X.shape)
X_train=X
y_train=y
# st.write(X_train.shape)
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor #direct multioutput wrapper
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
y_scale = 1e-5
y_train= y_train*y_scale

#model
model = MultiOutputRegressor(model())
model.fit(X_train, y_train)


context_len, pred_len = 20, 20
jump = 20
X, y = split_sequences(df, context_len, pred_len, jump, 'num_orders')
X = X.reshape(X.shape[0], -1) #Flatten time and dim for ML models. 107, 200
X_train=X[:]
X_train = sc.transform(X_train)
y_train_pred = model.predict(X_train).reshape(-1)
# st.write(y_train_pred.shape)


jump=pred_len
X = np.array(df)
# X_train = X
X_test = X[-context_len:].reshape(1,-1)
# st.write(X[-context_len:].shape)
# sc.transform(X_train)
sc.transform(X_test)
# st.write(X_train.shape, X_test.shape)

# y_train_pred = model.predict(X_train)

# X_test = sc.transform(df.iloc[:training_days-context_len])
y_test_pred = model.predict(X_test)
# st.write(y_test_pred.shape)

fig=plt.figure(figsize=(20,6), dpi=150)
plt.plot(np.array(df['num_orders']), color='royalblue', linewidth=5, label='Actual Time Series')
plt.fill_between(np.arange(np.array(df['num_orders']).shape[0]), np.array(df['num_orders']), color='royalblue',alpha=0.1)
plt.plot(np.arange(context_len, context_len+y_train_pred.shape[0]), 1/y_scale*y_train_pred , color='orangered', linewidth=3, label='Train Pred')


# st.write(X[-1,6])
prd = np.concatenate([X[-1,6].reshape(-1), 1/y_scale* y_test_pred.reshape(-1)])[:forecasting_days]
plt.plot(np.arange(X.shape[0], X.shape[0]+prd.shape[0]), prd , color='orange', linewidth=3, label='Actual Time Series')
plt.fill_between(np.arange(X.shape[0], X.shape[0]+prd.shape[0]), prd , color='orange', alpha=0.1)
plt.legend()
plt.title("Forecasting plot")
st.pyplot(fig)



if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)