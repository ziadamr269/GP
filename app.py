from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
file_path = 'twitter_training.csv'  # Update with the correct path
data = pd.read_csv(file_path, header=None)

# Rename columns
data.columns = ['id', 'game', 'sentiment', 'comment']

# Fill NaN values in the comments with an empty string
data['comment'] = data['comment'].fillna('')

# Encode sentiment labels
label_encoder = LabelEncoder()
data['sentiment_encoded'] = label_encoder.fit_transform(data['sentiment'])

# Split the data into training and testing sets
X = data['comment']
y = data['sentiment_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize the comments
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict on the test set
y_pred = model.predict(X_test_vec)

# Evaluate the model
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


@app.route('/predict', methods=['POST'])
def predict():
    new_comments = request.json.get('comments', [])
    if not new_comments:
        return jsonify({"error": "No comments provided"}), 400

    new_comments_vec = vectorizer.transform(new_comments)
    predictions = model.predict(new_comments_vec)
    predicted_labels = label_encoder.inverse_transform(predictions)

    results = [{"comment": comment, "sentiment": label} for comment, label in zip(new_comments, predicted_labels)]

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
