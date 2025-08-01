import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score


st.set_page_config(page_title="Titanic Survival Model", layout="wide")
st.title('Titanic Survival Model')
st.write('Working with titanic dataset')

data = pd.read_csv("https://raw.githubusercontent.com/miku-taj/titanic_survival_model/refs/heads/master/Clean-Titanic-Dataset.csv")

# Ideas - generate random person and predic function
# Randomly picking a person from the dataset
# Or like when showing a sample - add a button that renews the sample 
# But first - dataset familiarization: shape, info, columns meaning, then plots
# allow user to upload csv rather than thru buttons - but then we need to implement outputs for a whole bunch of instances?
# take into account user might not know some values yk? like, fare
# Side bar menu with proper styling
# mb represent the predictin graphically somehow? 

custom_css = """
    <style>
    a {
        text-decoration: none;
        color: #000000;
    }
    a:hover {
        text-decoration: underline
    }
    </style>
    """

st.sidebar.markdown('''
# Sections
- [Dataset shape](#dataset-shape)
- [Random 10 rows](#random-10-rows)
- [Visualization](#visualization)
''', unsafe_allow_html=True)


st.header("Dataset shape")
st.write(f"Rows: {data.shape[0]} Columns: {data.shape[1]}")

st.header("Random 10 rows")
st.dataframe(data.sample(10), use_container_width=True)

st.header("Visualization")

sns.set_theme(style="whitegrid", palette="Set2", font_scale=0.6)
fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(2, 3)

ax1 = fig.add_subplot(gs[0, :2])
ax1.set_title('Возраст пассажиров и выживших')
ax1.set_xlabel('Возраст')
ax1.set_ylabel('Кол-во')
sns.histplot(data=data, x='Age', hue='Survived', multiple='stack', ax=ax1, alpha=1.0)

ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('Сиблинги/супруги и выживание')
ax2.set_xlabel('Братья/Сестры/Супруги ')
ax2.set_ylabel('Кол-во')
sns.countplot(x='SibSp', hue='Survived', data=data, ax=ax2, alpha=1.0)

ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title('Дети/родители и выживание')
ax3.set_xlabel('Дети/Родители ')
ax3.set_ylabel('Кол-во')
sns.countplot(x='Parch', hue='Survived', data=data, ax=ax3, alpha=1.0)

ax4 = fig.add_subplot(gs[0, 2])
ax4.set_title('Класс пассажира и выживание')
ax4.set_xlabel('Класс')
ax4.set_ylabel('Кол-во')
sns.countplot(x='Pclass', hue='Survived', data=data, ax=ax4, alpha=1.0)

ax5 = fig.add_subplot(gs[1, 2])
ax5.set_title('Порт посадки и выживание')
ax5.set_xlabel('Порт')
ax5.set_ylabel('Кол-во')
sns.countplot(x='Embarked', hue='Survived', data=data, ax=ax5, alpha=1.0)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)

X = data.drop('Survived', axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

encoder = TargetEncoder(cols=['Name Prefix', 'Sex', 'Embarked'])
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X_train_scaled, y_train)
y_predict = model.predict(X_test_scaled)


# results = []
# for name, model in models.items():
#   model.fit(X_train_encoded, y_train)
#   acc_train = accuracy_score(y_train, model.predict(X_train_encoded))
#   acc_test = accuracy_score(y_test, model.predict(X_test_encoded))
#   results.append({
#     'Model': name,
#     'Train Accuracy': round(acc_train, 2),
#     'Test Accuracy': round(acc_test, 2)
#   })

# st.subheader("Comparing models metrics")
# st.table(pd.DataFrame(results))

# st.sidebar.header("Prediction based on features")

st.header('Make Prediction')

# island_input = st.sidebar.selectbox("Island", df['island'].unique())
# sex_input = st.sidebar.selectbox("Sex", df['sex'].unique())
# bill_length_input = st.sidebar.slider("Bill Length (mm)", float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()), float(df['bill_length_mm'].mean()))
# bill_depth_input = st.sidebar.slider("Bill Depth (mm)", float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()), float(df['bill_depth_mm'].mean()))
# flipper_length_input = st.sidebar.slider("Flipper Length (mm)", float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()), float(df['flipper_length_mm'].mean()))
# body_mass_input = st.sidebar.slider("Body Mass (g)", float(df['body_mass_g'].min()), float(df['body_mass_g'].max()), float(df['body_mass_g'].mean()))


with st.form("user_input_form"):
    # Categorical inputs
    pclass_input = st.selectbox("Класс (Pclass)", [1, 2, 3], index=2)
    sex_input = st.radio("Пол (Sex)", ['male', 'female'])
    embarked_input = st.selectbox("Порт посадки (Embarked)", ['C', 'Q', 'S'])
    prefix_input = st.selectbox("Обращение (Name Prefix)", ['Mr', 'Mrs', 'Miss', 'Master', 'Dr', 'Rev', 'Other'])

    # Numerical inputs
    age_input = st.number_input("Возраст (Age)", min_value=0.0, max_value=100.0, step=1.0)
    sibsp_input = st.number_input("Братья/сестры или супруг(а) на борту (SibSp)", min_value=0, max_value=10, step=1)
    parch_input = st.number_input("Родители/дети на борту (Parch)", min_value=0, max_value=10, step=1)
    fare_input = st.number_input("Плата за билет (Fare)", min_value=0.0, max_value=600.0, step=1.0)

    submitted = st.form_submit_button("📊 Предсказать")



user_input = pd.DataFrame([{
  'Pclass': pclass_input,
  'Sex': sex_input,
  'Age': age_input,
  'SibSp': sibsp_input,
  'Parch': parch_input,
  'Fare': fare_input,
  'Embarked': embarked_input,
  'Name Prefix': prefix_input
}])

user_input_encoded = encoder.transform(user_input)

for col in ['Age', 'SibSp', 'Parch', 'Fare']:
  user_input_encoded[col] = user_input[col].values
    
user_input_scaled = scaler.transform(user_input_encoded)
user_input_scaled = user_input_scaled[X_train_scaled.columns]


# st.sidebar.subheader("Prediction Results")

# for name, model in models.items():
#   pred = model.predict(user_input_encoded)[0]
#   proba = model.predict_proba(user_input_encoded)[0]
#   st.sidebar.markdown(f"**{name}: {pred}**")
#   proba_df = pd.DataFrame({'Species': model.classes_, 'Probability': proba})
#   st.sidebar.dataframe(proba_df, use_container_width=True)
