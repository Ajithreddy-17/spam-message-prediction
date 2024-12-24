# Spam Detection Project

This project demonstrates the implementation of a spam detection system using Python. The dataset used is a collection of SMS messages, labeled as `spam` or `ham`. The model is trained using a logistic regression classifier with TF-IDF vectorization to classify text messages as spam or non-spam.

---

## Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.x
- Required Python libraries:
  - pandas
  - scikit-learn
  - numpy

You can install the necessary packages using pip:
```bash
pip install pandas scikit-learn numpy
```

---

## Dataset

The dataset used in this project is a CSV file named `spam.csv` which contains two main columns:
- `v1`: Labels (`spam` or `ham`)
- `v2`: Text messages

The script renames these columns to `label` and `text` for convenience and filters out unnecessary columns.

---

## Workflow

1. **Data Preprocessing**
    - Load the CSV dataset using pandas.
    - Rename columns (`v1` to `label`, `v2` to `text`).
    - Map `spam` and `ham` labels to numerical values (`spam` = 1, `ham` = 0).
    - Check for and handle missing values (if any).

2. **Train-Test Split**
    - Split the dataset into training and testing sets using an 80-20 split.

3. **TF-IDF Vectorization**
    - Convert text messages into numerical representations using `TfidfVectorizer`.
    - Apply vectorization to both training and testing data.

4. **Model Training**
    - Train a logistic regression model on the vectorized training data.

5. **Model Evaluation**
    - Evaluate the model on the test data using metrics like accuracy, confusion matrix, and classification report.

6. **Prediction on New Messages**
    - Predict whether a new message is spam or not using the trained model.

---

## Code Walkthrough

### Import Libraries
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

### Data Preprocessing
```python
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data.rename(columns={"v1": "label", "v2": "text"})
data = data[['label', 'text']]
data['label'] = data['label'].map({'spam': 1, 'ham': 0})
```

### Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)
```

### TF-IDF Vectorization
```python
vectorizer = TfidfVectorizer(min_df=5, max_df=0.7, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

### Model Training and Evaluation
```python
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

### Predicting New Messages
```python
new_message = ["Congratulations! You've won a $1,000 gift card. Click here to claim."]
new_message_tfidf = vectorizer.transform(new_message)
print("Prediction:", model.predict(new_message_tfidf))
```

---

## Results

The model achieves high accuracy on the test data:
- **Accuracy:** ~96.5%
- **Precision, Recall, and F1-Score:** Evaluated for both `spam` and `ham` classes.

Example prediction:
- Input: `"Congratulations! You've won a $1,000 gift card. Click here to claim."`
- Output: `1` (spam)

---

## File Structure

```
project-directory/
|-- spam.csv           # Dataset file
|-- spam_detection.py  # Main script
|-- README.md          # Documentation
```

---

## How to Run

1. Clone the repository.
2. Place the `spam.csv` dataset in the project directory.
3. Run the script:
   ```bash
   python spam_detection.py
   ```
4. Observe the output metrics and predictions.

---

## Future Improvements

- Add support for other machine learning models like Naive Bayes or SVM.
- Incorporate hyperparameter tuning for better model optimization.
- Implement a front-end interface for real-time predictions.
- Explore deep learning approaches using LSTMs or transformers for improved accuracy.

---

## License

This project is licensed under the MIT License. Feel free to use and modify the code.

