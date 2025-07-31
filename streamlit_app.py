# import streamlit as st
# import pandas as pd
# import numpy as np
# import category_encoders as ce
# import plotly.express as px
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier

# st.set_page_config(page_title="🐧 Penguin Classifier", layout="wide")
# st.title('🐧 Penguin Classifier')
# st.write('Working with penguin dataset')

# df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")

# st.subheader("Dataset shape")
# st.write("Rows: ", df.shape[0])
# st.write("Columns: ", df.shape[1])

# st.subheader("🔍 Random 10 rows")
# st.dataframe(df.sample(10), use_container_width=True)

# st.subheader("🔍 Visualization")
# col1, col2 = st.columns(2)
# with col1:
#   fig1 = px.histogram(df, x="species", color="island", barmode="group", title="Distribution of species across islands")
#   st.plotly_chart(fig1, use_container_width=True)
# with col2:
#   fig2 = px.scatter(df, x="bill_length_mm", y="flipper_length_mm", color="species", title="Bill length vs Flipper length")
#   st.plotly_chart(fig2, use_container_width=True)

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
