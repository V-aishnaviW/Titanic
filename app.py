
import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing tools
model = joblib.load("titanic_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

categorical_cols = ['Sex', 'Embarked', 'Cabin', 'Pclass']

st.title("🚢 Titanic Survival Predictor")

Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 1, 80, 25)
SibSp = st.slider("Siblings/Spouses Aboard", 0, 5, 0)
Parch = st.slider("Parents/Children Aboard", 0, 5, 0)
Fare = st.slider("Fare", 0.0, 600.0, 50.0)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
Cabin = st.text_input("Cabin", "Unknown")

if st.button("Predict"):
    df = pd.DataFrame([{
        "Pclass": Pclass,
        "Sex": Sex,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Embarked": Embarked,
        "Cabin": Cabin
    }])

    df['Cabin'] = df['Cabin'].apply(lambda x: x[0] if x else 'U')
    encoded = pd.DataFrame(encoder.transform(df[categorical_cols]), columns=encoder.get_feature_names_out())
    df.drop(categorical_cols, axis=1, inplace=True)
    df = pd.concat([df, encoded], axis=1)
    df[['Age', 'Fare']] = scaler.transform(df[['Age', 'Fare']])
    pred = model.predict(df)[0]
    st.success("✅ Survived" if pred == 1 else "❌ Did Not Survive")
