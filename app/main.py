from flask import Flask, render_template, send_file
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.tree import plot_tree
import joblib
import os

app = Flask(__name__)

# Load the dataset
data_path = os.path.join('app', 'smoking.csv')
data = pd.read_csv(data_path)

# Convert categorical columns to numerical for correlation analysis
categorical_columns = ['gender', 'oral', 'tartar']
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Calculate the correlation matrix
correlation_matrix = data_encoded.corr()

# Focus on the correlation with the target variable 'smoking'
correlation_with_smoking = correlation_matrix['smoking'].sort_values(ascending=False)

# Select top 13 features with the highest correlation
top_features = correlation_with_smoking.head(14).index.tolist()
if 'smoking' in top_features:
    top_features.remove('smoking')

# Create a new dataset with selected features and the target variable
data_selected = data_encoded[top_features + ['smoking']]

# Split the data into training and testing sets
X = data_selected.drop('smoking', axis=1)
y = data_selected['smoking']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Save the model
model_path = os.path.join('app', 'random_forest_smoking_model.pkl')
joblib.dump(model, model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/features')
def features():
    return {'Top Features': top_features}

@app.route('/correlation-matrix')
def correlation_matrix_plot():
    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Features')
    plt.savefig('static/correlation_matrix.png')
    return send_file('static/correlation_matrix.png', mimetype='image/png')

@app.route('/roc-curve')
def roc_curve_plot():
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('static/roc_curve.png')
    return send_file('static/roc_curve.png', mimetype='image/png')

@app.route('/evaluation-metrics')
def evaluation_metrics():
    accuracy = accuracy_score(y_test, model.predict(X_test))
    precision = precision_score(y_test, model.predict(X_test))
    recall = recall_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test))
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    return metrics

@app.route('/tree-graph')
def tree_graph():
    plt.figure(figsize=(20, 10))
    plot_tree(model.estimators_[0], feature_names=X.columns, filled=True, rounded=True, max_depth=3)
    plt.title('Random Forest Tree (Max Depth=3)')
    plt.savefig('static/tree_graph.png')
    return send_file('static/tree_graph.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
