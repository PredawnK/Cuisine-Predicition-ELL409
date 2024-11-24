import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def load_data(train_path, test_path):
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    return train_data, test_data

train_data, test_data = load_data('/kaggle/input/dataset/train.json', '/kaggle/input/dataset/test.json')

for recipe in train_data:
    for idx, ingredient in enumerate(recipe["ingredients"]):
        recipe["ingredients"][idx] = ingredient.replace(" ", "_")

for recipe in test_data:
    for idx, ingredient in enumerate(recipe["ingredients"]):
        recipe["ingredients"][idx] = ingredient.replace(" ", "_")


unique_ingredients = set()

for recipe in train_data:
    for ingredient in recipe["ingredients"]:
        unique_ingredients.add(ingredient)  # Add each ingredient to the set

# The number of unique ingredients
num_features = len(unique_ingredients)
print("num_features = ", num_features)


# Prepare the data for TF-IDF
train_texts = [' '.join(recipe['ingredients']) for recipe in train_data]
test_texts = [' '.join(recipe['ingredients']) for recipe in test_data]


# Apply TF-IDF
vectorizer = TfidfVectorizer(max_features=num_features)  # Limit to top 5000 features for efficiency
X_train = vectorizer.fit_transform(train_texts).toarray()
X_test = vectorizer.transform(test_texts).toarray()

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform([recipe['cuisine'] for recipe in train_data])


# Apply SMOTE for class balancing if necessary
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)




# After applying SMOTE, #(samples) for all the classes increased to 5910
# count = {}
# for y in y_train_resampled:
#     if y in count: count[y] += 1
#     else: count[y] = 1

# print(count)



class CuisinePredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CuisinePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        # self.fc4 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        # x = self.relu(self.fc4(x))
        x = self.softmax(self.fc3(x))
        return x

input_dim = X_train_resampled.shape[1]
output_dim = len(label_encoder.classes_)
model = CuisinePredictor(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_resampled, y_train_resampled, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


train_accuracies = []
val_accuracies = []


# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_preds = []
    train_labels = []
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_accuracy = accuracy_score(train_labels, train_preds)
    train_accuracies.append(train_accuracy)

    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, '
          f'Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, '
          f'Val Accuracy: {val_accuracy:.4f}')

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy', marker='o')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Prepare the test data tensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# Predict cuisines for the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()

# Convert predicted labels back to original cuisine names
test_ids = [recipe['id'] for recipe in test_data]
test_cuisines = label_encoder.inverse_transform(test_preds)

# Create submission file
submission = pd.DataFrame({'Id': test_ids, 'Category': test_cuisines})
submission.to_csv('submission.csv', index=False)