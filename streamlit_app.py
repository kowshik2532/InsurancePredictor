import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import openai

# Set OpenAI API Key
openai.api_key = "your-api-key-here"  # ❗Replace with your actual API key or use environment variables for security.

# Page config
st.set_page_config(page_title="PremiumPredict", layout="wide")
st.title("🏥 Health Insurance Premium Predictor")

# File upload
uploaded_file = st.file_uploader("📂 Browse your CSV file", type=["csv"])

# Initialize session state
for key in ["show_summary", "show_eda", "show_custom_vis", "build_model", "chat_with_your_ds"]:
    if key not in st.session_state:
        st.session_state[key] = False

# Data cleaning function
def clean_data(df):
    df_cleaned = df.copy()
    df_cleaned.dropna(inplace=True)
    df_cleaned.columns = [col.strip().replace(" ", "_").lower() for col in df_cleaned.columns]
    return df_cleaned

# If file is uploaded
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.session_state.df_raw = df_raw
    st.write("### 📄 Data Overview")
    st.dataframe(df_raw.head(), use_container_width=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("📋 Data Summary"):
            st.session_state.update({"show_summary": True, "show_eda": False, "show_custom_vis": False, "build_model": False, "chat_with_your_ds": False})
    with col2:
        if st.button("📈 Run EDA"):
            st.session_state.update({"show_summary": False, "show_eda": True, "show_custom_vis": False, "build_model": False, "chat_with_your_ds": False})
    with col3:
        if st.button("🎨 Visualize your data"):
            st.session_state.update({"show_summary": False, "show_eda": False, "show_custom_vis": True, "build_model": False, "chat_with_your_ds": False})
    with col4:
        if st.button("🧠 Build ML Model"):
            st.session_state.update({"show_summary": False, "show_eda": False, "show_custom_vis": False, "build_model": True, "chat_with_your_ds": False})
    with col5:
        if st.button("💬 Chat with your Dataset"):
            st.session_state.update({"show_summary": False, "show_eda": False, "show_custom_vis": False, "build_model": False, "chat_with_your_ds": True})

    # Summary
    if st.session_state.show_summary:
        st.subheader("🔎 Raw Dataset Overview")
        st.markdown(f"- **Rows:** {df_raw.shape[0]}")
        st.markdown(f"- **Columns:** {df_raw.shape[1]}")

        st.subheader("🧱 Data Types")
        st.dataframe(df_raw.dtypes.reset_index().rename(columns={"index": "Column", 0: "Data Type"}), use_container_width=True)

        st.subheader("🧩 Missing Values")
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.dataframe(missing.reset_index().rename(columns={"index": "Column", 0: "Missing Values"}), use_container_width=True)
        else:
            st.info("✅ No missing values detected.")

        st.subheader("📊 Descriptive Statistics")
        st.dataframe(df_raw.describe(include='all'), use_container_width=True)

        if st.button("🧼 Clean Data"):
            df_cleaned = clean_data(df_raw)
            st.session_state.df_cleaned = df_cleaned
            st.success("✅ Data cleaned successfully!")
            st.write("### 🔍 Cleaned Data Preview")
            st.dataframe(df_cleaned.head(), use_container_width=True)

    # ML Model
    if st.session_state.build_model:
        st.subheader("🧠 Build Machine Learning Model")

        if "df_cleaned" not in st.session_state:
            st.info("🔄 Cleaning data before model building...")
            df_cleaned = clean_data(st.session_state.df_raw)
            st.session_state.df_cleaned = df_cleaned
            st.success("✅ Data cleaned successfully!")
        else:
            df_cleaned = st.session_state.df_cleaned

        target_col = st.selectbox("🎯 Select Target Column", df_cleaned.columns)

        # Check if target is continuous or categorical
        if df_cleaned[target_col].dtype in ['float64', 'int64']:
            task_type = "Regression"
        else:
            task_type = "Classification"

        st.write(f"#### 🧠 Task Type: {task_type}")

        # If regression, select regression model; if classification, select classification model
        if task_type == "Regression":
            algo_choice = st.selectbox("🤖 Choose ML Algorithm", ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"])
        else:
            algo_choice = st.selectbox("🤖 Choose ML Algorithm", ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier"])

        # Feature selection
        if df_cleaned[target_col].dtype in ['float64', 'int64']:
            corr_matrix = df_cleaned.select_dtypes(include='number').corr()
            corr_with_target = corr_matrix[target_col].abs().sort_values(ascending=False)
            corr_with_target = corr_with_target[corr_with_target < 1.0]
            top_features = corr_with_target[:int(0.75 * len(corr_with_target))].index.tolist()
        else:
            st.warning("Correlation only works for numeric targets. Using all numeric features.")
            numeric_cols = df_cleaned.select_dtypes(include='number').columns.drop(target_col, errors='ignore')
            top_features = numeric_cols.tolist()

        st.markdown("#### 🔝 Top 75% Most Correlated Features:")
        st.write(top_features)

        X = df_cleaned[top_features]
        y = df_cleaned[target_col]

        # Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if st.button("🚀 Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            if algo_choice == "Linear Regression":
                model = LinearRegression()
            elif algo_choice == "Decision Tree Regressor":
                model = DecisionTreeRegressor()
            elif algo_choice == "Random Forest Regressor":
                model = RandomForestRegressor()
            elif algo_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif algo_choice == "Decision Tree Classifier":
                model = DecisionTreeClassifier()
            elif algo_choice == "Random Forest Classifier":
                model = RandomForestClassifier()

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            st.subheader("📉 Model Performance")

            if task_type == "Regression":
                st.success(f"✅ R² (R-squared): {model.score(X_test, y_test):.4f}")

            else:
                accuracy = accuracy_score(y_test, predictions)
                st.success(f"✅ Accuracy: {accuracy:.4f}")

            # Add input box for a new prediction
            st.subheader("🔮 Make a Prediction")
            input_data = []
            for feature in top_features:
                value = st.number_input(f"Enter value for {feature}", value=0.0)
                input_data.append(value)

            if st.button("🔍 Predict"):
                input_scaled = scaler.transform([input_data])
                prediction = model.predict(input_scaled)

                if task_type == "Regression":
                    st.write(f"🔮 Predicted Target Value: {prediction[0]:.4f}")
                else:
                    st.write(f"🔮 Predicted Target Class: {prediction[0]}")
