
import os
import re
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import metrics, preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from alive_progress import alive_bar


target_classes = ["go", "stop", "warning"]
color_map = {"go" : "green", "stop" : "red", "warning" : "yellow"}
rgb_color_map = {"go" : (0, 255, 0), "stop": (255, 0, 0), "warning": (255, 255, 0)}

#total de imagens utilizadas para normalizar o dataset
n_samples_per_class = 1000

train_folder_list = [
    "dayTrain",
    "nightTrain"
]

def ler_dataset(cenarios_teste):
    caminho_dataset = "dataset"
    caminho_annotation = "dataset" + os.sep + "Annotations"

    annotation_list = list()
    for folder in cenarios_teste:
        caminho_annotation_folder = caminho_annotation + os.sep + folder

        df = pd.DataFrame()
        if 'Clip' in os.listdir(caminho_annotation_folder)[0]:
            clip_list = os.listdir(caminho_annotation_folder)
            for clip_folder in clip_list:
                df = pd.read_csv(caminho_annotation_folder + os.sep + clip_folder + os.sep + "frameAnnotationsBOX.csv", sep=";")
                df["image_path"] = caminho_dataset + os.sep  + folder + os.sep + clip_folder + os.sep + "frames" + os.sep
                annotation_list.append(df)

        df = pd.concat(annotation_list)
        df = df.drop(['Origin file', 'Origin frame number', 'Origin track', 'Origin track frame number'], axis=1)
        df.columns = ['filename', 'target', 'x1', 'y1', 'x2', 'y2', 'image_path']
        df = df[df['target'].isin(target_classes)]
        df['filename'] = df['filename'].apply(lambda filename: re.findall("\/([\d\w-]*.jpg)", filename)[0])
        df = df.drop_duplicates().reset_index(drop=True)
    return df

def crop_semaforo(df):
    print("extraindo semaforos")
    img_values = dict()
    with alive_bar(len(df) + 1) as bar:
        index = 0
        bar(index, skipped=False)
        for index, row in df.iterrows():
            image_path = row["image_path"]
            name = row["filename"]
            x1, x2, y1, y2 = row["x1"], row["x2"], row["y1"], row["y2"]
            print(image_path + name)
            img = cv2.imread(image_path + name)
            cropped_img = img[y1:y2, x1:x2]
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            img_values[index] = cropped_img
            bar()
    return img_values


def undersample_dataset(annotation_df, n_samples):
    print("Fazendo Undersampling do Dataset")
    df_resample_list = list()
    for target in target_classes:
        df = annotation_df[annotation_df['target'] == target].copy()
        df_r = resample(df, n_samples=n_samples, random_state=42)
        df_resample_list.append(df_r)
    return pd.concat(df_resample_list).reset_index(drop=True)


def bin_images(img_values):
    print("Binarizando imagens...")
    binary_img_values = dict()
    with alive_bar(len(img_values) + 1) as bar:
        index = 0
        bar(index, skipped=False)
        for index in range(0,len(img_values)):
            img = img_values[index]
            img = cv2.resize(img, (30, 50))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, binary_img = cv2.threshold(gray, 0, 255, 
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            binary_img[binary_img == 0] = 1
            binary_img[binary_img == 255] = 0
            binary_img_values[index] = binary_img
            bar()
    return binary_img_values


def convert_to_vector(binary_img_values, annotation_df):
    print("Convertendo imagens para vetor...")
    binary_img_vectors = dict()
    with alive_bar(len(img_values)) as bar:
        for index in range(0, len(binary_img_values)):
            img = binary_img_values[index]
            img_vector = np.c_[img.ravel()].transpose()[0]
            binary_img_vectors[index] = img_vector
            bar()

    img_columns = ["p"+str(i) for i in range(1,1501)]
    df_img_values = pd.DataFrame(binary_img_vectors.values(), 
                                 index=binary_img_vectors.keys(), 
                                 columns=img_columns)
    return pd.merge(annotation_df, df_img_values, left_index=True, right_index=True)


if __name__ == "__main__":
    train_annotation_df = ler_dataset(train_folder_list)

    target_classes = train_annotation_df['target'].unique()
    target_classes.sort()
    
    index, counts = np.unique(train_annotation_df['target'], return_counts=True)
    colors = [color_map[target] for target in index]
    plt.bar(index, counts, color=colors)
    plt.legend(loc="best")
    plt.title('Total de tipos de Imagens no Dataset')
    plt.show()

    train_annotation_df = undersample_dataset(train_annotation_df, n_samples_per_class)
    index, counts = np.unique(train_annotation_df['target'], return_counts=True)
    colors = [color_map[target] for target in index]
    plt.bar(index, counts, color=colors)
    plt.legend(loc="best")
    plt.title('Total de tipos de Imagens no Após Undersampling (1000 Amostras)')
    plt.show()

    img_values = crop_semaforo(train_annotation_df)
    
    samples_imgs = dict()
    for target in target_classes:
        index = train_annotation_df[train_annotation_df['target'] == target].index[0]
        img = img_values[index]
        samples_imgs[target] = img
   
    binary_img_values = bin_images(img_values)

    samples_imgs = dict()
    for target in target_classes:
        index = train_annotation_df[train_annotation_df['target'] == target].index[0]
        img = binary_img_values[index]
        samples_imgs[target] = img

    df_img_values = convert_to_vector(binary_img_values, train_annotation_df)

    print("Treinando Classificador...")

    X = df_img_values.drop(['filename', 'target', 'x1', 'y1', 'x2', 'y2', 'image_path'], axis=1)
    y = df_img_values["target"]
    pca = PCA(n_components=2)
    X_r = pca.fit_transform(X)
    y_r = pd.Series(y.values.ravel())

    plt.figure(figsize=(10,8))
    for target in target_classes:
        plt.scatter(X_r[y_r == target, 0], 
                    X_r[y_r == target, 1], 
                    color=color_map[target], alpha=.8, label=target)
    plt.legend(loc="best")
    plt.title('PCA')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    print("Testando Classificador - Reporte do SVM")
    svc = SVC()
    svc_train = svc.fit(X_train, y_train.values.ravel())
    y_pred = svc.predict(X_test)
    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_estimator(
    svc_train, X_test, y_test)
    plt.show()
    