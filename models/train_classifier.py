import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import warnings

warnings.simplefilter("ignore")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def load_data(database_filepath):
    """
    Loads the data from the database filepath.
    Args:
        database_filepath: The path to the database file.

    Returns:
        X: The messages used to train the model.
        Y: The labels for the messages.
        category_names: The names of the categories.
    """

    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("DisasterResponse", engine)
    X = df["message"]
    Y = df.drop(["id", "message", "original", "genre"], axis=1)
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes the text.
    Args:
        text: The text to tokenize.
    Returns:
        tokens: The tokens of the text.
    """

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [
        lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words("english")
    ]

    return clean_tokens


def build_model(tuning=False):
    """
    Builds the model.
    Args:
        tuning: If true, will use the GridSearchCV.
                If false, will use the pipeline directly.
    Returns:
        model: The model.
    """

    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )

    if tuning:
        parameters = [
            {
                "clf": [RandomForestClassifier()],
                "clf__n_estimators": [100, 250],
            },
            {
                "clf": [KNeighborsClassifier()],
                "clf__weights": ["uniform", "distance"],
            },
            {
                "clf": [GaussianNB()],
            },
            {
                "clf": [LinearSVC()],
                "clf__C": [1.0, 10.0],
            },
        ]

        model = GridSearchCV(pipeline, param_grid=parameters, refit=True, verbose=3)
    else:
        model = pipeline

    return model


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluates the model.
    Args:
        model: The model.
        X_test: The test messages.
        Y_test: The test labels.
        category_names: The names of the categories.
    returns:
        dataframe: The dataframe with the evaluation
            metrics:precision, recall, f1-score
    """
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=y_test.columns)
    results = pd.DataFrame([])
    for column in category_names:
        report = precision_recall_fscore_support(
            y_test[column], y_pred[column], beta=1, average="weighted"
        )
        result = (
            pd.DataFrame(report)
            .transpose()
            .rename(columns={0: "precision", 1: "recall", 2: "f1-score", 3: "support"})
        )
        result["target"] = column
        results = pd.concat([results, result])
    results.set_index("target", inplace=True)
    results = results.drop("support", axis=1)
    print(results)

    return results


def save_model(model, model_filepath):
    """
    Saves the model.
    Args:
        model: The model.
        model_filepath: The path to the model file.
    returns:
        None
    """
    # Save the model to disk
    pickle.dump(model, open(model_filepath, "wb"))

    return


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
