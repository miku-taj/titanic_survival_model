import streamlit as st
import pandas as pd
import numpy as np
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Titanic Survival Model", layout="wide")
st.title('Titanic Survival Model')
st.write('Working with titanic dataset')

data = pd.read_csv("https://raw.githubusercontent.com/miku-taj/titanic_survival_model/refs/heads/master/Titanic-Dataset.csv")

# Ideas - generate random person and predic function
# Randomly picking a person from the dataset
# Or like when showing a sample - add a button that renews the sample 
# But first - dataset familiarization: shape, info, columns meaning, then plots
# allow user to upload csv rather than thru buttons - but then we need to implement outputs for a whole bunch of instances?


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

# sns.set(font_scale=0.8)
fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(2, 3)

ax1 = fig.add_subplot(gs[0, :2])
ax1.set_title('Возраст пассажиров и выживших')
ax1.set_label('Возраст')
ax1.set_label('Кол-во')
sns.histplot(data=data, x='Age', hue='Survived', multiple='stack', ax=ax1)

ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('Сиблинги/супруги и выживание')
ax2.set_xlabel('Братья/Сестры/Супруги ')
ax2.set_ylabel('Кол-во')
sns.countplot(x='SibSp', hue='Survived', data=data, ax=ax2)

ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title('Дети/родители и выживание')
ax3.set_xlabel('Дети/Родители ')
ax3.set_ylabel('Кол-во')
sns.countplot(x='Parch', hue='Survived', data=data, ax=ax3)

ax4 = fig.add_subplot(gs[0, 2])
ax4.set_title('Класс пассажира и выживание')
ax4.set_xlabel('Класс')
ax4.set_ylabel('Кол-во')
sns.countplot(x='Pclass', hue='Survived', data=data, ax=ax4)

ax5 = fig.add_subplot(gs[1, 2])
ax5.set_title('Порт посадки и выживание')
ax5.set_xlabel('Порт')
ax5.set_ylabel('Кол-во')
sns.countplot(x='Embarked', hue='Survived', data=data, ax=ax5)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)

# X = df.drop(["species"], axis=1)
# y = df["species"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# encoder = ce.TargetEncoder(cols=['island', 'sex'])
# X_train_encoded = encoder.fit_transform(X_train, y_train)
# X_test_encoded = encoder.transform(X_test)

# models = {
#   'Decision Tree': DecisionTreeClassifier(random_state=42),
#   'KNN': KNeighborsClassifier()
# }

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
# island_input = st.sidebar.selectbox("Island", df['island'].unique())
# sex_input = st.sidebar.selectbox("Sex", df['sex'].unique())
# bill_length_input = st.sidebar.slider("Bill Length (mm)", float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()), float(df['bill_length_mm'].mean()))
# bill_depth_input = st.sidebar.slider("Bill Depth (mm)", float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()), float(df['bill_depth_mm'].mean()))
# flipper_length_input = st.sidebar.slider("Flipper Length (mm)", float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()), float(df['flipper_length_mm'].mean()))
# body_mass_input = st.sidebar.slider("Body Mass (g)", float(df['body_mass_g'].min()), float(df['body_mass_g'].max()), float(df['body_mass_g'].mean()))

# user_input = pd.DataFrame([{
#   'island': island_input,
#   'sex': sex_input,
#   'bill_length_mm': bill_length_input,
#   'bill_depth_mm': bill_depth_input,
#   'flipper_length_mm': flipper_length_input,
#   'body_mass_g': body_mass_input
# }])

# user_input_encoded = encoder.transform(user_input)
# for col in ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']:
#   user_input_encoded[col] = user_input[col].values
# user_input_encoded = user_input_encoded[X_train_encoded.columns]

# st.sidebar.subheader("Prediction Results")

# for name, model in models.items():
#   pred = model.predict(user_input_encoded)[0]
#   proba = model.predict_proba(user_input_encoded)[0]
#   st.sidebar.markdown(f"**{name}: {pred}**")
#   proba_df = pd.DataFrame({'Species': model.classes_, 'Probability': proba})
#   st.sidebar.dataframe(proba_df, use_container_width=True)
