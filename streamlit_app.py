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

df = pd.read_csv("https://raw.githubusercontent.com/miku-taj/titanic_survival_model/refs/heads/master/Titanic-Dataset.csv")

# Ideas - generate random person and predic function
# Randomly picking a person from the dataset
# Or like when showing a sample - add a button that renews the sample 
# But first - dataset familiarization: shape, info, columns meaning, then plots
# allow user to upload csv rather than thru buttons - but then we need to implement outputs for a whole bunch of instances?

st.subheader("Dataset shape")
st.write(f"Rows: {df.shape[0]} Columns: {df.shape[1]}")

st.subheader("🔍 Random 10 rows")
st.dataframe(df.sample(10), use_container_width=True)

st.subheader("🔍 Visualization")
# col1, col2 = st.columns(2)
# with col1:
#   fig1 = px.histogram(df, x="species", color="island", barmode="group", title="Distribution of species across islands")
#   st.plotly_chart(fig1, use_container_width=True)
# with col2:
#   fig2 = px.scatter(df, x="bill_length_mm", y="flipper_length_mm", color="species", title="Bill length vs Flipper length")
#   st.plotly_chart(fig2, use_container_width=True)

# Create a figure
fig = plt.figure(figsize=(10, 15))

# Define the GridSpec: 2 rows, 2 columns
# The top subplot will span both columns of the first row
# The bottom two subplots will each occupy one column of the second row
gs = fig.add_gridspec(3, 2)

# First subplot (top, spans both columns)
ax1 = fig.add_subplot(gs[0, :]) # gs[0, :] means first row, all columns
ax1.set_title('Распределение возраста среди всех пассажиров и выживших пассажиров')
ax1.set_label('Возраст')
ax1.set_label('Кол-во')
sns.histplot(data=data, x='Age', hue='Survived', multiple='stack', ax=ax1)

# Second subplot (bottom-left)
ax2 = fig.add_subplot(gs[1, 0]) # gs[1, 0] means second row, first column
ax2.set_title('Кол-во братьев/сестр/супруга/супруги')
ax2.set_xlabel('Братья/Сестры/Супруги ')
ax2.set_ylabel('Кол-во')
sns.barplot(data=data.groupby('SibSp').count(), x='SibSp', y='PassengerId', ax=ax2)
sns.barplot(data=data.groupby('SibSp').sum(), x='SibSp', y='Survived', ax=ax2, color='orange')

# Third subplot (bottom-right)
ax3 = fig.add_subplot(gs[1, 1]) # gs[1, 1] means second row, second column
ax3.set_title('Кол-во детей/родителей')
ax3.set_xlabel('Дети/Родители ')
ax3.set_ylabel('Кол-во')
sns.barplot(data=data.groupby('Parch').count(), x='Parch', y='PassengerId', ax=ax3)
sns.barplot(data=data.groupby('Parch').sum(), x='Parch', y='Survived', ax=ax3, color='orange')

ax4 = fig.add_subplot(gs[2, 0]) # gs[1, 1] means second row, second column
ax4.set_title('Число пассажиров и выживших разных классов')
ax4.set_xlabel('Класс')
ax4.set_ylabel('Кол-во')
sns.barplot(data=data.groupby('Pclass').count(), x='Pclass', y='PassengerId', ax=ax4)
sns.barplot(data=data.groupby('Pclass').sum(), x='Pclass', y='Survived', ax=ax4, color='orange')

ax5 = fig.add_subplot(gs[2, 1]) # gs[1, 1] means second row, second column
ax5.set_title('Кол-во пассажиров и выживших с разных портов посадки')
ax5.set_xlabel('Порт')
ax5.set_ylabel('Кол-во')
sns.barplot(data=data.groupby('Embarked').count(), x='Embarked', y='PassengerId', ax=ax5)
sns.barplot(data=data.groupby('Embarked').sum(), x='Embarked', y='Survived', ax=ax5, color='orange')

plt.tight_layout()
st.plotly_chart(fig, use_container_width=True)

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
