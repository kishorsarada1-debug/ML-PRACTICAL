import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Iris Flower Classification App")
st.write(
    "Enter flower measurements to predict whether an Iris flower "
    "is Setosa, Versicolor, or Virginica."
)
st.divider()

# -----------------------------------------
# 2. LOAD DATASET
# -----------------------------------------
@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    data = iris.frame.copy()
    data["species"] = data["target"].map(
        dict(enumerate(iris.target_names))
    )
    data = data.drop(columns=["target"])
    return data, iris.feature_names, iris.target_names

df, feature_names, target_names = load_data()

st.header("📊 Iris Dataset")
st.write("The app uses the built-in Iris dataset from scikit-learn.")
st.dataframe(df.head(10), use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", len(df))
col2.metric("Features", len(feature_names))
col3.metric("Flower Classes", len(target_names))

# -----------------------------------------
# 3. DATA VISUALIZATION
# -----------------------------------------
st.header("📈 Explore the Dataset")
plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="sepal length (cm)",
        y="sepal width (cm)",
        hue="species",
        ax=ax
    )
    ax.set_title("Sepal Measurements by Species")
    st.pyplot(fig)
    plt.close(fig)

with plot_col2:
    fig, ax = plt.subplots()
    sns.boxplot(
        data=df,
        x="species",
        y="petal length (cm)",
        ax=ax
    )
    ax.set_title("Petal Length by Species")
    ax.set_xlabel("Species")
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------------------
# 4. TRAIN MACHINE LEARNING MODEL
# -----------------------------------------
st.header("🤖 Machine Learning Model")

X = df[feature_names]
y = load_iris().target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)

st.metric("Test Accuracy", f"{accuracy:.2%}")

with st.expander("View model evaluation"):
    report = classification_report(
        y_test,
        test_predictions,
        target_names=target_names,
        output_dict=True
    )
    st.dataframe(
        pd.DataFrame(report).transpose().round(3),
        use_container_width=True
    )

    matrix = confusion_matrix(y_test, test_predictions)
    fig, ax = plt.subplots()
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------------------
# 5. USER INPUT AND PREDICTION
# -----------------------------------------
st.divider()
st.header("🔍 Predict an Iris Flower")
st.write("Adjust the measurements below, then click Predict.")

left, right = st.columns(2)

with left:
    sepal_length = st.slider(
        "Sepal length (cm)", 4.0, 8.0, 5.8, 0.1
    )
    sepal_width = st.slider(
        "Sepal width (cm)", 2.0, 4.5, 3.0, 0.1
    )

with right:
    petal_length = st.slider(
        "Petal length (cm)", 1.0, 7.0, 4.0, 0.1
    )
    petal_width = st.slider(
        "Petal width (cm)", 0.1, 2.5, 1.2, 0.1
    )

if st.button("🌸 Predict Flower Species", type="primary"):
    input_values = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=feature_names
    )

    predicted_class = int(model.predict(input_values)[0])
    probabilities = model.predict_proba(input_values)[0]
    predicted_species = target_names[predicted_class]

    st.subheader("Prediction Result")
    st.success(f"Predicted species: **{predicted_species.title()}**")

    st.write("Prediction probabilities:")
    probability_df = pd.DataFrame({
        "Species": [name.title() for name in target_names],
        "Probability": probabilities
    }).sort_values("Probability", ascending=False)

    st.bar_chart(
        probability_df.set_index("Species"),
        y="Probability"
    )

    st.caption(
        "This is a demonstration model trained on the classic Iris dataset. "
        "Predictions depend on the measurements entered."
    )

# -----------------------------------------
# 6. FOOTER
# -----------------------------------------
st.divider()
st.caption(
    "Iris Flower Classification | Python • Streamlit • Scikit-learn"
)
