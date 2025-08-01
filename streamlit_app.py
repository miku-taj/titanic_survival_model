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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay
)



st.set_page_config(page_title="Модель выживания на Титанике", layout="wide")
st.title("Модель выживания на Титанике")
st.write("Работа с данными пассажиров Титаника")

# if 'prediction_button_clicked' not in st.session_state:
#     st.session_state.prediction_button_clicked = False

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
# Разделы
- [Размер датасета](#razmer-dataseta)
- [Случайные 10 строк](#sluchaynye-10-strok)
- [Визуализация](#vizualizatsiya)
- [Метрики модели](#metriki-modeli)
- [Сделать прогноз](#sdelat-prognoz)
''', unsafe_allow_html=True)


st.header("Размер датасета")
st.write(f"Rows: {data.shape[0]} Columns: {data.shape[1]}")

st.header("Случайные 10 строк")
st.dataframe(data.sample(10), use_container_width=True)

st.header("Визуализация")

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

data_columns = set(data.columns)
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

st.header('Метрики модели')

def compute_metrics(model, X_train, y_train, X_test, y_test):

    result = {'Train': {},
              'Test': {},
              'Train-Test Difference': {}}
    metrics = {
        "Accuracy": accuracy_score,
        "Precision": precision_score,
        "Recall": recall_score,
        "F1 Score": f1_score,
        "ROC AUC": roc_auc_score
        }

    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:, 1]

    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    for name, metric in metrics.items():
        train_score = metric(y_train, y_pred_train)
        test_score = metric(y_test, y_pred_test)
        result['Train'][name] = train_score
        result['Test'][name] = test_score
        result['Train-Test Difference'][name] = train_score - test_score

    return pd.DataFrame(result)

st.table(compute_metrics(model, X_train_scaled, y_train, X_test_scaled, y_test))

matrix = plt.figure(figsize=(5, 4))
# sns.set(font_scale=1.1, style="whitegrid")
cm = confusion_matrix(y_test, model.predict(X_test_scaled))
sns.heatmap(cm, annot=True, cmap="Blues", cbar=False,
            xticklabels=["Predicted No", "Predicted Yes"],
            yticklabels=["Actual No", "Actual Yes"])
plt.title("Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.tight_layout()
st.pyplot(matrix)

# st.subheader("Comparing models metrics")
# st.table(pd.DataFrame(results))



st.header('Сделать прогноз')

with st.form("user_input_form"):

    pclass_input = st.selectbox("Класс (Pclass)", list(data['Pclass'].unique()), index=0)
    sex_input = st.radio("Пол (Sex)", ['male', 'female'])
    embarked_input = st.selectbox("Порт посадки (Embarked)", list(data['Embarked']), index=2)
    prefix_input = st.selectbox("Обращение (Name Prefix)", list(data['Name Prefix'].unique()), index=0)

    age_input = st.number_input("Возраст (Age)", min_value=0, max_value=100, value=int(data['Age'].median()), step=1)
    sibsp_input = st.number_input("Братья/сестры или супруг(а) на борту (SibSp)", min_value=int(data['SibSp'].min()), max_value=int(data['SibSp'].max()), step=1)
    parch_input = st.number_input("Родители/дети на борту (Parch)", min_value=int(data['Parch'].min()), max_value=int(data['Parch'].max()), step=1)
    fare_input = st.slider("Плата за билет (Fare)", min_value=float(data['Fare'].min()), max_value=float(data['Fare'].max()), value=float(data['Fare'].mean()))
    submit_button = st.form_submit_button("Предсказать")

    if submit_button:
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
        
        with st.expander('See results below'):
            pred = model.predict(user_input_scaled)[0]
            proba = model.predict_proba(user_input_encoded)[0]
            if pred == 1:
                st.markdown("**Сожалеем, этот человек погиб на Титанике.**")
            else:
                st.markdown("**Поздравляем, этот человек выжил на Титанике!**")


st.header("Или загрузите CSV-файл")
st.info("Вы можете загрузить CSV-файл с данными пассажиров Титаника.")

with st.container():
    uploaded_file = st.file_uploader("Выберите файл", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            user_csv = pd.read_csv(uploaded_file)
            if data_columns.issubset(set(user_csv.columns)):
                st.success("Файл успешно загружен!")
            else:
                st.error(f"Файл не содержит необходимых столбцов")
        except Exception as e:
            st.error(f"Ошибка при чтении файла: {e}")
    else:
        st.caption("Файл ещё не загружен.")











st.header("Prediction Results")

            
# for name, model in models.items():
#   pred = model.predict(user_input_encoded)[0]
#   proba = model.predict_proba(user_input_encoded)[0]
#   st.sidebar.markdown(f"**{name}: {pred}**")
#   proba_df = pd.DataFrame({'Species': model.classes_, 'Probability': proba})
#   st.sidebar.dataframe(proba_df, use_container_width=True)
