import cv2
import mahotas as mh
import pandas as pd
from torchvision import transforms

def extract_texture_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    textures = mh.features.haralick(gray_image.astype(int), return_mean=True)

    idm = textures[4]

    contrast = textures[1]

    entropy = textures[8]

    return idm, entropy, contrast

def preprocess_and_save_features(image_paths, save_path):
    features = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        idm, entropy, contrast = extract_texture_features(image)
        features.append([image_path, idm, entropy, contrast])

    df = pd.DataFrame(features, columns=['image_path', 'idm', 'entropy', 'contrast'])
    df.to_csv(save_path, index=False)
    print(f'Features saved to {save_path}')

def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
