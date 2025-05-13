import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import openai
import os

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Use environment variable for security

# Page config
st.set_page_config(page_title="PremiumPredict", layout="wide")
st.title("🏥 Health Insurance Premium Predictor")

# File upload
uploaded_file = st.file_uploader("📂 Browse your CSV file", type=["csv"])

# Initialize session state
for key in ["show_summary", "show_eda", "show_custom_vis", "build_model", "chat_with_your_ds", "show_predict_form"]:
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
        st.dataframe(df_raw.dtypes.to_frame(name="Data Type").reset_index().rename(columns={"index": "Column"}), use_container_width=True)

        st.subheader("🧩 Missing Values")
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.dataframe(missing.reset_index().rename(columns={"index": "Column", 0: "Missing Values"}), use_container_width=True)
        else:
            st.info("✅ No missing values detected.")

        st.subheader("📊 Descriptive Statistics")
        st.dataframe(df_raw.describe(include='all').transpose(), use_container_width=True)

        if st.button("🧼 Clean Data"):
            df_cleaned = clean_data(df_raw)
            st.session_state.df_cleaned = df_cleaned
            st.success("✅ Data cleaned successfully!")
            st.write("### 🔍 Cleaned Data Preview")
            st.dataframe(df_cleaned.head(), use_container_width=True)

    # EDA
    if st.session_state.show_eda:
        st.subheader("🔍 Exploratory Data Analysis (EDA)")
        categorical_cols = df_raw.select_dtypes(include='object').columns
        st.markdown("#### 🎯 Categorical Variable Distributions")
        for col in categorical_cols:
            st.markdown(f"**{col}**")
            st.bar_chart(df_raw[col].value_counts())

        st.markdown("#### 📌 Correlation Matrix (Numeric Columns)")
        corr = df_raw.select_dtypes(include='number').corr()
        if not corr.empty:
            st.dataframe(corr, use_container_width=True)
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("No numeric columns to compute correlation.")

        st.markdown("#### 📈 Histograms for Numeric Columns")
        for col in df_raw.select_dtypes(include='number').columns:
            fig, ax = plt.subplots()
            sns.histplot(df_raw[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            plt.tight_layout()
            st.pyplot(fig)

    # Custom Viz
    if st.session_state.show_custom_vis:
        st.subheader("🎨 Custom Data Visualization")
        col_opts = df_raw.columns.tolist()
        x_axis = st.selectbox("Select X-axis:", col_opts)
        y_axis = st.selectbox("Select Y-axis (optional):", ["None"] + col_opts)
        chart_type = st.selectbox("Choose Chart Type:", ["Scatter", "Bar", "Line", "Histogram", "Boxplot"])

        if st.button("📊 Visualize"):
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            try:
                if chart_type == "Scatter":
                    if y_axis != "None":
                        sns.scatterplot(data=df_raw, x=x_axis, y=y_axis, ax=ax)
                    else:
                        st.error("Scatter plot requires both X and Y.")
                elif chart_type == "Bar":
                    if y_axis != "None":
                        sns.barplot(data=df_raw, x=x_axis, y=y_axis, ax=ax)
                    else:
                        df_raw[x_axis].value_counts().plot(kind="bar", ax=ax)
                elif chart_type == "Line":
                    if y_axis != "None":
                        sns.lineplot(data=df_raw, x=x_axis, y=y_axis, ax=ax)
                    else:
                        st.error("Line plot requires both X and Y.")
                elif chart_type == "Histogram":
                    sns.histplot(df_raw[x_axis].dropna(), kde=True, ax=ax)
                elif chart_type == "Boxplot":
                    if y_axis != "None":
                        sns.boxplot(data=df_raw, x=x_axis, y=y_axis, ax=ax)
                    else:
                        sns.boxplot(data=df_raw, y=x_axis, ax=ax)

                ax.set_title(f"{chart_type} of {x_axis}" + (f" vs {y_axis}" if y_axis != "None" else ""))
                plt.tight_layout()

                # Place the figure inside a centered column to control width
                col1, col2, col3 = st.columns([1, 6, 1])
                with col2:
                    st.pyplot(fig, clear_figure=True)

            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")



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

        algo_choice = st.selectbox("🤖 Choose ML Algorithm", ["Linear Regression", "Decision Tree", "Random Forest", "Logistic Regression"])

        if st.button("🚀 Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            is_classification = y.nunique() <= 10 and y.dtype in ['int64', 'object']

            if algo_choice == "Linear Regression":
                model = LinearRegression()
            elif algo_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif algo_choice == "Decision Tree":
                model = DecisionTreeClassifier() if is_classification else DecisionTreeRegressor()
            elif algo_choice == "Random Forest":
                model = RandomForestClassifier() if is_classification else RandomForestRegressor()

            model.fit(X_train, y_train)
            st.session_state.model = model
            st.session_state.X_columns = X.columns.tolist()

            st.success("✅ Model trained successfully!")

            st.subheader("📊 Model Evaluation")
            y_pred = model.predict(X_test)

            if isinstance(model, (LinearRegression, DecisionTreeRegressor, RandomForestRegressor)):
                r2 = r2_score(y_test, y_pred)
                st.write(f"R-squared (R²): {r2:.4f}")
            else:
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy:.4f}")

        if "model" in st.session_state:
            if st.button("Predict a custom case"):
                st.session_state.show_predict_form = True

            if st.session_state.get("show_predict_form", False):
                st.subheader("🔍 Predict on Custom Input")
                input_data = {}
                for feature in st.session_state.X_columns:
                    val = st.text_input(f"Enter value for {feature}", value="0")
                    try:
                        input_data[feature] = float(val)
                    except:
                        input_data[feature] = val

                if st.button("🧪 Predict"):
                    input_df = pd.DataFrame([input_data])
                    try:
                        prediction = st.session_state.model.predict(input_df)[0]
                        st.success(f"🧠 Predicted Value: {prediction}")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
        else:
            st.info("⚠️ Train a model first before making predictions.")

    # Chat with dataset
    if st.session_state.chat_with_your_ds:
        st.subheader("💬 Chat with your Dataset")
        user_input = st.text_input("Ask your dataset anything...")

        if user_input:
            if openai.api_key is None:
                st.error("⚠️ OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
            else:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful data analyst."},
                            {"role": "user", "content": user_input}
                        ],
                        max_tokens=300,
                        temperature=0.5
                    )
                    st.write("### Response from AI:")
                    st.write(response['choices'][0]['message']['content'].strip())
                except Exception as e:
                    st.error(f"Error while processing your query: {str(e)}")
