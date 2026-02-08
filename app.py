"""
Twitter Sentiment Analysis - Streamlit App
NLP: tokenization → stemming → lemmatization → TF-IDF
Models: Naive Bayes, Logistic Regression, Decision Tree, Random Forest
"""
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from nlp_pipeline import pipeline_for_vectorizer
from load_data import load_dataset, get_sample_data

# Page config
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1da1f2;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #1da1f2;
        padding-bottom: 0.5rem;
    }
    .sub-header { color: #657786; font-size: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #f7f9fc 0%, #e8f4fc 100%);
        padding: 1rem 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #1da1f2;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .stButton > button {
        background: linear-gradient(135deg, #1da1f2 0%, #0d8bd9 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.25rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0d8bd9 0%, #0a6ba8 100%);
        box-shadow: 0 4px 12px rgba(29, 161, 242, 0.4);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #15202b 0%, #192734 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown { color: #e7e9ea; }
    .positive { color: #17bf63; font-weight: 600; }
    .negative { color: #e0245e; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(use_custom_path=None):
    """Load dataset (cached)."""
    return load_dataset(use_custom_path)


def get_models():
    """Return dict of model name -> sklearn classifier."""
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }


def train_and_evaluate(df, vectorizer, models):
    """
    Train all 4 models, return metrics, fitted vectorizer, fitted models,
    y_true, y_pred per model, and best model name.
    """
    # Binary: 0 -> 0, 4 -> 1 (Sentiment140)
    y = (df["target"].values == 4).astype(int)
    X_text = df["text"].astype(str).values

    # NLP pipeline + TF-IDF
    X_tfidf = vectorizer.fit_transform(X_text)
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = (y_test, y_pred)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
        })

    best_name = max(results, key=lambda x: x["F1"])["Model"]
    metrics_df = pd.DataFrame(results)
    return metrics_df, vectorizer, models, predictions, best_name, X_test, y_test


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix with seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax,
        cbar_kws={"label": "Count"},
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=14, fontweight=600)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    return fig


def main():
    st.markdown('<p class="main-header">🐦 Twitter Sentiment Analysis</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">NLP pipeline: Tokenization → Stemming → Lemmatization → TF-IDF | '
        'Models: Naive Bayes, Logistic Regression, Decision Tree, Random Forest</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        st.markdown("---")
        data_source = st.radio(
            "Dataset",
            ["Built-in sample (labeled tweets)", "Custom CSV file"],
            index=0,
        )
        custom_path = None
        if data_source == "Custom CSV file":
            custom_path = st.text_input(
                "Path to CSV (e.g. data/training.csv)",
                value="data/training.csv",
            )
        train_clicked = st.button("🔄 Train models", type="primary", use_container_width=True)
        st.markdown("---")
        st.caption("After training, use the tabs for comparison, confusion matrix, live detection, and export.")

    # Load data
    df = load_data(custom_path if data_source == "Custom CSV file" else None)
    st.sidebar.caption(f"Dataset: {len(df):,} tweets")

    # Session state for trained artifacts
    if "metrics_df" not in st.session_state:
        st.session_state.metrics_df = None
    if "vectorizer" not in st.session_state:
        st.session_state.vectorizer = None
    if "models" not in st.session_state:
        st.session_state.models = None
    if "best_model_name" not in st.session_state:
        st.session_state.best_model_name = None
    if "predictions" not in st.session_state:
        st.session_state.predictions = None

    if train_clicked:
        with st.spinner("Running NLP pipeline and training 4 models..."):
            vectorizer = TfidfVectorizer(
                preprocessor=pipeline_for_vectorizer,
                tokenizer=lambda x: x.split(),
                max_features=10000,
                ngram_range=(1, 2),
            )
            models = get_models()
            metrics_df, vec, fitted_models, predictions, best_name, _, _ = train_and_evaluate(
                df, vectorizer, models
            )
            st.session_state.metrics_df = metrics_df
            st.session_state.vectorizer = vec
            st.session_state.models = fitted_models
            st.session_state.best_model_name = best_name
            st.session_state.predictions = predictions
        st.sidebar.success("Training complete.")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Model comparison",
        "📈 Confusion matrix",
        "✏️ Live sentiment",
        "📝 Demo sample tweets",
        "📥 Export predictions",
    ])

    with tab1:
        st.subheader("Model comparison")
        if st.session_state.metrics_df is not None:
            comparison = st.session_state.metrics_df.copy()
            for col in ["Accuracy", "Precision", "Recall", "F1"]:
                comparison[col] = comparison[col].round(4)
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.caption("Best model (by F1): **" + st.session_state.best_model_name + "**")
        else:
            st.info("Click **Train models** in the sidebar to see the comparison table.")

    with tab2:
        st.subheader("Confusion matrix (best model)")
        if st.session_state.predictions is not None and st.session_state.best_model_name is not None:
            best = st.session_state.best_model_name
            y_true, y_pred = st.session_state.predictions[best]
            fig = plot_confusion_matrix(y_true, y_pred, title=f"Confusion Matrix — {best}")
            st.pyplot(fig)
            plt.close(fig)
            st.markdown(f"**Classification report — {best}**")
            report = classification_report(
                y_true, y_pred, target_names=["Negative", "Positive"], zero_division=0
            )
            st.code(report, language="text")
        else:
            st.info("Train models first to see the confusion matrix.")

    with tab3:
        st.subheader("Live sentiment detection")
        if st.session_state.vectorizer is not None and st.session_state.models is not None:
            model_choice = st.selectbox(
                "Model to use",
                list(st.session_state.models.keys()),
                index=list(st.session_state.models.keys()).index(st.session_state.best_model_name)
                if st.session_state.best_model_name else 0,
            )
            user_text = st.text_area("Enter or paste text to analyze", height=120, placeholder="Type a tweet or any text here...")
            if st.button("Analyze sentiment"):
                if user_text.strip():
                    vec = st.session_state.vectorizer
                    model = st.session_state.models[model_choice]
                    X = vec.transform([user_text])
                    pred = model.predict(X)[0]
                    proba = getattr(model, "predict_proba", lambda X: None)(X)
                    if proba is not None:
                        prob = proba[0][1]
                    else:
                        prob = 1.0 if pred == 1 else 0.0
                    label = "Positive" if pred == 1 else "Negative"
                    st.markdown(f"**Sentiment:** <span class=\"{'positive' if pred == 1 else 'negative'}\">{label}</span> (confidence: {prob:.2%})", unsafe_allow_html=True)
                else:
                    st.warning("Please enter some text.")
        else:
            st.info("Train models first to use live sentiment detection.")

    with tab4:
        st.subheader("Demo on sample tweets")
        if st.session_state.vectorizer is not None and st.session_state.models is not None:
            demo_tweets = [
                "I love this new feature! So helpful.",
                "Worst experience ever. Never again.",
                "It's okay, nothing special.",
                "Amazing support team. Fixed my issue in minutes!",
                "Terrible quality. Broke after one day.",
            ]
            model_choice = st.selectbox(
                "Model",
                list(st.session_state.models.keys()),
                key="demo_model",
            )
            vec = st.session_state.vectorizer
            model = st.session_state.models[model_choice]
            demo_df = pd.DataFrame({"Tweet": demo_tweets})
            X_demo = vec.transform(demo_tweets)
            preds = model.predict(X_demo)
            demo_df["Predicted sentiment"] = ["Positive" if p == 1 else "Negative" for p in preds]
            st.dataframe(demo_df, use_container_width=True, hide_index=True)
        else:
            st.info("Train models first to run the demo.")

    with tab5:
        st.subheader("Export predictions as CSV")
        if st.session_state.predictions is not None and st.session_state.metrics_df is not None:
            model_choice = st.selectbox(
                "Model",
                list(st.session_state.predictions.keys()),
                key="export_model",
            )
            y_true, y_pred = st.session_state.predictions[model_choice]
            export_df = pd.DataFrame({
                "true_label": ["Positive" if y == 1 else "Negative" for y in y_true],
                "predicted_label": ["Positive" if y == 1 else "Negative" for y in y_pred],
            })
            buf = io.StringIO()
            export_df.to_csv(buf, index=False)
            st.download_button(
                "Download predictions (CSV)",
                data=buf.getvalue(),
                file_name="sentiment_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Train models first to export predictions.")

    # Data preview in sidebar
    with st.sidebar:
        st.markdown("---")
        with st.expander("Preview data"):
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
