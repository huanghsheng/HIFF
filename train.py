import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score
from config import *
from utils import preprocess_and_save_features
from dataset import CustomDataset
from model import HIFF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

labels_df = pd.read_excel(TRAIN_PRO_PATH)
y = labels_df.iloc[:, 0].values

if not os.path.exists(FEATURE_CSV_PATH):
    preprocess_and_save_features(IMAGE_PATHS, FEATURE_CSV_PATH)

X_train_paths, X_test_paths, y_train, y_test = train_test_split(IMAGE_PATHS, y, test_size=TRAIN_TEST_SPLIT[1], random_state=42)
X_train_CLAHE_paths, X_test_CLAHE_paths = train_test_split(IMAGE_CLAHE_PATHS, test_size=TRAIN_TEST_SPLIT[1], random_state=42)

train_dataset = CustomDataset(X_train_paths, X_train_CLAHE_paths, y_train, FEATURE_CSV_PATH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = HIFF().to(device)

class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

num_epochs = NUM_EPOCHS
best_train_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    for images, textures, labels in train_loader:
        images, textures, labels = images.to(device), textures.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, textures)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    epoch_loss = epoch_loss / len(train_loader)
    accuracy = 100 * correct / total
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    if accuracy > best_train_accuracy:
        best_train_accuracy = accuracy
        torch.save(model.state_dict(), 'best_train_model.pth')
        print(f'Saved best model at epoch {epoch+1} with training accuracy: {best_train_accuracy:.2f}%')

    scheduler.step()

torch.save(model.state_dict(), 'final_model.pth')