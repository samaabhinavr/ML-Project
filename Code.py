import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import StepLR

# Load the dataset
data = pd.read_csv("matches.csv")

# Data cleaning and preprocessing
data_cleaned = data.drop(columns=['Unnamed: 0', 'date', 'time', 'comp', 'round', 'day', 'match report', 'notes', 'season', 'team'])
data_cleaned = data_cleaned.dropna()

# Encode categorical variables
label_encoder = LabelEncoder()
data_cleaned['venue'] = label_encoder.fit_transform(data_cleaned['venue'])  # Home=0, Away=1
data_cleaned['result'] = label_encoder.fit_transform(data_cleaned['result'])  # W=2, D=1, L=0

# Feature engineering: Team form and metrics
def calculate_team_form(df):
    df = df.copy()
    df['win'] = (df['result'] == 2).astype(int)  # Win=2
    df['goal_diff'] = df['gf'] - df['ga']
    
    # Rolling averages for team form
    df['avg_gf'] = df['gf'].rolling(window=3, min_periods=1).mean().shift(1)
    df['avg_ga'] = df['ga'].rolling(window=3, min_periods=1).mean().shift(1)
    df['avg_goal_diff'] = df['goal_diff'].rolling(window=3, min_periods=1).mean().shift(1)
    df['win_rate'] = df['win'].rolling(window=3, min_periods=1).mean().shift(1)

    # Recent form score: Win=3, Draw=1, Loss=0
    df['recent_form'] = df['result'].replace({2: 3, 1: 1, 0: 0}).rolling(window=3, min_periods=1).mean().shift(1)
    
    # Additional team-based features
    df['recent_conceded'] = df['ga'].rolling(window=3, min_periods=1).mean().shift(1)
    df['recent_scored'] = df['gf'].rolling(window=5, min_periods=1).mean().shift(1)
    return df

# Apply feature engineering
data_cleaned = calculate_team_form(data_cleaned)
data_cleaned = data_cleaned.dropna()

# Features and target
X = data_cleaned[['venue', 'avg_gf', 'avg_ga', 'avg_goal_diff', 'win_rate', 'recent_form', 'recent_conceded', 'recent_scored']]
y = data_cleaned['result']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

# Define the neural network model
class FootballPredictor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(FootballPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Hyperparameter optimization
def train_and_evaluate_model(X_tensor, y_tensor, hidden_size1, hidden_size2, lr, batch_size):
    input_size = X_tensor.shape[1]
    output_size = len(np.unique(y_tensor.numpy()))
    model = FootballPredictor(input_size, hidden_size1, hidden_size2, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    conf_matrices = []
    
    for train_idx, val_idx in skf.split(X_tensor, y_tensor):
        X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
        y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
        
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(30):  # Reduced epochs for CV
            model.train()
            for inputs, labels in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            _, predicted = torch.max(outputs, 1)
            accuracy = accuracy_score(y_val, predicted)
            accuracies.append(accuracy)
            conf_matrices.append(confusion_matrix(y_val, predicted))
    
    return np.mean(accuracies), conf_matrices, model

# Hyperparameter grid search
hidden_size1_list = [64, 128]
hidden_size2_list = [32, 64]
lr_list = [0.001, 0.0005]
batch_size_list = [16, 32]

best_accuracy = 0
best_params = {}
final_model = None
confusion_matrices = []

for h1 in hidden_size1_list:
    for h2 in hidden_size2_list:
        for lr in lr_list:
            for batch_size in batch_size_list:
                accuracy, conf_matrices, model = train_and_evaluate_model(X_tensor, y_tensor, h1, h2, lr, batch_size)
                print(f"Params: h1={h1}, h2={h2}, lr={lr}, batch_size={batch_size} -> Accuracy: {accuracy:.4f}")
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {'hidden_size1': h1, 'hidden_size2': h2, 'lr': lr, 'batch_size': batch_size}
                    final_model = model
                    confusion_matrices = conf_matrices

print("Best Params:", best_params)
print(f"Best Cross-Validation Accuracy: {best_accuracy:.2f}%")

# Aggregate and plot confusion matrices
final_conf_matrix = sum(confusion_matrices)
plt.figure(figsize=(8, 6))
sns.heatmap(final_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Draw', 'Win'], yticklabels=['Loss', 'Draw', 'Win'])
plt.title('Final Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('final_confusion_matrix.png')
plt.show()

# Feature Importance
feature_importance = final_model.fc1.weight.data.abs().mean(dim=0).numpy()
feature_names = ['venue', 'avg_gf', 'avg_ga', 'avg_goal_diff', 'win_rate', 'recent_form', 'recent_conceded', 'recent_scored']

plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('final_feature_importance.png')
plt.show()

# Final Analysis
print("Final Model Analysis:")
print("- Cross-Validation Accuracy:", best_accuracy)
print("- Confusion Matrix and Feature Importance plots saved.")
