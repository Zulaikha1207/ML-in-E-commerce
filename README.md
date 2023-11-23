## Ecommerce Classifier

### Description
The Ecommerce Classifier is a Python-based machine learning project that predicts the category of products on an e-commerce platform. The logistic regression model uses product titles, reviews, and ratings as input features to classify products into predefined categories.

### Features
- Data Integration: Product data and review data are integrated and variations in column order and misspelled categories is taken care of

- Data Preprocessing: Simple preprocessing steps, including converting text to lowercase, removing special characters

- Data Validation: Ensures the integrity and quality of the input data. It checks for input DataFrame's type, presence of samples, existence of text and numerical columns, and numeric data type for rating

- Text Vectorization: TF-IDF vectorization is used to transform textual features, such as product titles and reviews, into numerical representations for machine learning

- Model Training and Evaluation: Logistic regression model is trained and evaluated on a test set, providing metrics such as accuracy, f1-score, and a detailed classification report

- Cross-Validation: The project includes cross-validation functionality, to give a better estimate of teh model's performance

- Confusion Matrix Visualization: The confusion matrices generated during both model evaluation and cross-validation are visualized using Seaborn, aiding in the interpretation of model performance

- Reproducibility: The project is designed with reproducibility in mind, allowing others to recreate the same virtual environment and dependencies using the provided requirements.txt file.

### Installation

#### Clone the repository:
`git clone https://github.com/Zulaikha1207/ML-in-E-commerce.git`

#### Navigate to the project directory:
`cd ML-in-E-commerce`

#### Install dependencies:
`pip install -r requirements.txt`

#### Usage
Run the ecommerce script
`python src/ecommerce_analysis.py`
