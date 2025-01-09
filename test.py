import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

from model import HIFF
from dataset import CustomDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

labels_df = pd.read_excel('./data/train_3_pro.xlsx')
y = labels_df.iloc[:, 0].values

image_paths = ['./data/606_pro/' + str(i) + '.png' for i in range(409)]
image_CLAHE_paths = ['./data/606_pro_CLAHE_1/' + str(i) + '.png' for i in range(409)]

# 提取并保存特征
feature_csv_path = './data/texture_features.csv'
if not os.path.exists(feature_csv_path):  # 如果特征文件不存在，则提取并保存
    print('Feature file does not exist. Please run preprocess_and_save_features first.')

_, X_test_paths, _, y_test = train_test_split(image_paths, y, test_size=0.3, random_state=42)
_, X_test_CLAHE_paths = train_test_split(image_CLAHE_paths, test_size=0.3, random_state=42)

test_dataset = CustomDataset(X_test_paths, X_test_CLAHE_paths, y_test, feature_csv_path)
test_loader = DataLoader(test_dataset, batch_size=18, shuffle=False)

model = HIFF().to(device)
model.load_state_dict(torch.load('./xxxxxxx'))

class_weights = torch.tensor([2, 4, 6], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

def test_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for test_images, test_textures, test_labels in test_loader:
            test_images, test_textures, test_labels = test_images.to(device), test_textures.to(device), test_labels.to(device)
            outputs = model(test_images, test_textures)
            loss = criterion(outputs, test_labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
            all_labels.extend(test_labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}%, Recall: {recall:.4f}, F1 Score: {f1:.4f}')


if __name__ == "__main__":
    test_model(model, test_loader)
