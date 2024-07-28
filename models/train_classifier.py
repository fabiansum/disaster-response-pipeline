import sys

import joblib
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

nltk.download(["punkt", "wordnet"])


def load_data(database_filepath):
    """
    Load data from a SQLite database.

    Parameters:
    database_filepath (str): File path of the SQLite database.

    Returns:
    X (Series): Series containing the messages.
    y (DataFrame): DataFrame containing the categories.
    category_names (Index): Index object containing the category names.
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql("SELECT * FROM disaster_response", engine)
    X = df["message"]
    y = df.iloc[:, 4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    Tokenize and lemmatize the input text.

    Parameters:
    text (str): Text to be tokenized and lemmatized.

    Returns:
    clean_tokens (list): List of clean tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline and perform grid search.

    Returns:
    model (GridSearchCV): Grid search model object.
    """
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(estimator=RandomForestClassifier())),
        ]
    )
    parameters = {
        "clf__estimator__n_estimators": [50, 100, 200],
        "clf__estimator__min_samples_split": [2, 3, 4],
    }

    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model on test data and print the classification report for each category.

    Parameters:
    model: Trained model.
    X_test (DataFrame): Test features.
    Y_test (DataFrame): True labels for test data.
    category_names (list): List of category names.
    """
    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column)
        print(classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    """
    Save the trained model to a file.

    Parameters:
    model: Trained model.
    model_filepath (str): File path to save the model.
    """
    filename = model_filepath
    joblib.dump(model, open(filename, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
