import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from data import data_registry
class EcommerceClassifier:
    def __init__(self, file_path, product_data, review_data):
        self.file_path = Path(file_path)
        self.product_data = data_registry.product_data
        self.review_data = data_registry.review_data
        self.df = None
        self.labels = None
        self.numerical_cols = ["rating"]
        self.text_cols = ["title", "review"]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None

    def read_data(self):
        product_dfs = []
        for x in self.product_data:
            df = pd.read_csv(self.file_path / f"{x}.tsv", sep='\t', names=['id', 'category', 'title'])
            # Replace misspelled category
            df['category'] = df['category'].replace('Ktchen', 'Kitchen')
            product_dfs.append(df)
            log.info(f"Initializing product data file: {x}")
        product_df = pd.concat(product_dfs, ignore_index=True)

        review_dfs = []
        for x in self.review_data:
            column_names = ['rating', 'id', 'review'] if x == "reviews-2" else ['id', 'rating', 'review']
            df = pd.read_csv(self.file_path / f"{x}.tsv", sep='\t', header=None, names=column_names)
            review_dfs.append(df)
            log.info(f"Initializing review data file: {x}")
        review_df = pd.concat(review_dfs, ignore_index=True)

        self.df = pd.merge(product_df, review_df, on='id')
        self.labels = self.df.pop('category')

    def preprocess_data(self):
        # Cinvert to lowercase, remove spaces and punctuations
        self.df['review'] = self.df['review'].str.lower()
        self.df['review'] = self.df['review'].str.replace('<br />', '')
        self.df['review'] = self.df['review'].str.replace('!', '')

    def check_data(self):
        # Input dataframe checks
        assert isinstance(self.df, pd.DataFrame), f"First input needs to be of type {pd.DataFrame}"
        assert len(self.df) > 0, "Some samples are required in the input DataFrame"

        # Check if text columns exist
        assert len([x for x in self.text_cols if x in self.df.columns]) > 0, "Some text data needed"

        # Check if numerical column exists
        assert len([x for x in self.numerical_cols if x in self.df.columns]) > 0, "Ratings data needed"

        # Check the data type of numerical column
        numerical_columns = [x for x in self.numerical_cols if x in self.df]
        assert all([is_numeric_dtype(self.df[x]) for x in numerical_columns]), "Ratings data needs to be numeric"

        # Check if the number of output classes is equal to 2
        assert len(self.labels.value_counts()) == 2, "Number of categories in 'labels' is not equal to 2"

    def split_data(self, test_size=0.2, random_state=42):
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df, self.labels, test_size=test_size, random_state=random_state)

    def transform_data(self):
        # Standardize numerical column, ratings
        scaler = StandardScaler()
        X_train_numerical_cols = scaler.fit_transform(self.X_train[self.numerical_cols])
        X_test_numerical_cols = scaler.transform(self.X_test[self.numerical_cols])

        # Fit vectorizer separately on title and review
        tfidf_vectorizer_title = TfidfVectorizer(stop_words='english')
        tfidf_matrix_title_train = tfidf_vectorizer_title.fit_transform(self.X_train[self.text_cols[0]])
        tfidf_matrix_title_test = tfidf_vectorizer_title.transform(self.X_test[self.text_cols[0]])

        tfidf_vectorizer_review = TfidfVectorizer(stop_words='english')
        tfidf_matrix_review_train = tfidf_vectorizer_review.fit_transform(self.X_train[self.text_cols[1]])
        tfidf_matrix_review_test = tfidf_vectorizer_review.transform(self.X_test[self.text_cols[1]])

        # Concatenate TF-IDF features with numerical columns
        self.X_train = pd.concat(
            [pd.DataFrame(tfidf_matrix_title_train.toarray(), columns=tfidf_vectorizer_title.get_feature_names_out()),
             pd.DataFrame(tfidf_matrix_review_train.toarray(), columns=tfidf_vectorizer_review.get_feature_names_out()),
             pd.DataFrame(X_train_numerical_cols, columns=self.numerical_cols)], axis=1)

        self.X_test = pd.concat(
            [pd.DataFrame(tfidf_matrix_title_test.toarray(), columns=tfidf_vectorizer_title.get_feature_names_out()),
             pd.DataFrame(tfidf_matrix_review_test.toarray(), columns=tfidf_vectorizer_review.get_feature_names_out()),
             pd.DataFrame(X_test_numerical_cols, columns=self.numerical_cols)], axis=1)

        self.label_encoder = LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(self.y_train)
        self.y_test = self.label_encoder.transform(self.y_test)

    def train_and_evaluate_model(self):
        logistic_regression_model = LogisticRegression()
        logistic_regression_model.fit(self.X_train, self.y_train)

        y_pred = logistic_regression_model.predict(self.X_test)

        accuracy_test = np.around(accuracy_score(self.y_test, y_pred), 2)
        log.info(f"Test Accuracy: {accuracy_test}")

        f1_test = np.round(f1_score(self.y_test, y_pred), 2)
        log.info(f"Test f1-score: {f1_test}")

        log.info(f"\n -------------Classification Report-------------\n")
        log.info(classification_report(self.y_test, y_pred))

        conf_matrix = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(5, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig("/Users/zulikahlatief/Desktop/personal/ecommerce/reports/figures/Validation Confusion Matrix", dpi=100)

    def cross_validate_model(self, num_splits=7):
        logistic_regression_model = LogisticRegression()
        kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
        y_pred_cv = cross_val_predict(logistic_regression_model, self.X_train, self.y_train, cv=kf)

        accuracy_cv = np.round(accuracy_score(self.y_train, y_pred_cv))
        log.info(f"Cross-Validation Accuracy: {accuracy_cv}")

        f1_test = np.round(f1_score(self.y_train, y_pred_cv), 2)
        log.info(f"Cross-Validation f1-score: {f1_test}")

        log.info(f"\n -------------Cross-Validation Classification Report-------------\n")
        log.info(classification_report(self.y_train, y_pred_cv))

        conf_matrix_cv = confusion_matrix(self.y_train, y_pred_cv)

        plt.figure(figsize=(5, 4))
        sns.heatmap(conf_matrix_cv, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Cross-Validation Confusion Matrix')
        #roc_plots_path = config['evaluate']['roc_plots_path']
        plt.savefig("/Users/zulikahlatief/Desktop/personal/ecommerce/reports/figures/Cross-Validation Confusion Matrix", dpi=100)

if __name__ == "__main__":
    ecommerce_classifier = EcommerceClassifier("/Users/zulikahlatief/Desktop/personal/ecommerce/src/data",
                                               product_data=data_registry.product_data,
                                               review_data=data_registry.review_data)

    ecommerce_classifier.read_data()
    ecommerce_classifier.preprocess_data()
    ecommerce_classifier.check_data()
    ecommerce_classifier.split_data()
    ecommerce_classifier.transform_data()
    ecommerce_classifier.train_and_evaluate_model()
    ecommerce_classifier.cross_validate_model()