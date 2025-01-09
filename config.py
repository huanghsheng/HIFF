import os

DATA_DIR = './data'
TRAIN_PRO_PATH = os.path.join(DATA_DIR, 'train_3_pro.xlsx')
FEATURE_CSV_PATH = os.path.join(DATA_DIR, 'texture_features.csv')
IMAGE_PATHS = [os.path.join(DATA_DIR, '606_pro', f'{i}.png') for i in range(409)]
IMAGE_CLAHE_PATHS = [os.path.join(DATA_DIR, '606_pro_CLAHE_1', f'{i}.png') for i in range(409)]

BATCH_SIZE = 18
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
STEP_SIZE = 8
GAMMA = 0.5
CLASS_WEIGHTS = [2, 4, 6]
TRAIN_TEST_SPLIT = [0.7, 0.3]

