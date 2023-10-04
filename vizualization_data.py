import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os
import sys

from MLandPattern import MLandPattern as ML

class_label = ["Non-target Language", "Target Language"]

alpha_val = 0.5


def load(pathname, vizualization=0):
    df = pd.read_csv(pathname, header=None)
    if vizualization:
        print(df.head())
    attribute = np.array(df.iloc[:, 0 : len(df.columns) - 1])
    attribute = attribute.T
    # print(attribute)
    label = np.array(df.iloc[:, -1])

    return attribute, label


def histogram_1n(male, female, x_axis="", y_axis=""):
    plt.xlim((min(min(male), min(female)) - 1, max(max(male), max(female)) + 1))
    plt.hist(male, color="blue", alpha=alpha_val, label=class_label[0], density=True, bins=np.arange(male.min(), male.max()+2,1), edgecolor='grey')
    plt.hist(
        female, color="red", alpha=alpha_val, label=class_label[1], density=True, bins=np.arange(female.min(), female.max()+2,1), edgecolor='grey'
    )
    plt.legend(class_label)





def graficar(attributes):
    attribute_names = []
    for i in range(attributes.shape[0]):
        attribute_names.append(str(i))
    values_histogram = {}

    for i in range(len(attribute_names)):
        values_histogram[attribute_names[i]] = [
            attributes[i, labels == 0],
            attributes[i, labels == 1],
        ]

    cont = 1

    for xk, xv in values_histogram.items():
        for yk, yv in values_histogram.items():
            if xk == yk:
                histogram_1n(xv[0], xv[1], x_axis=xk)

                plt.savefig(f"{os.getcwd()}/Image/histogram-dim-{cont}.png")
                plt.show()
                cont += 1


def graf_LDA(attributes, lables):

    W, _ = ML.LDA1(attributes, lables, 1)
    LDA_attributes = np.dot(W.T, attributes)
    print(LDA_attributes.shape)
    histogram_1n(LDA_attributes[0, labels==0], LDA_attributes[0, labels==1])
    plt.title("LDA Direction")
    plt.savefig(f"{os.getcwd()}/Image/LDA-direction.png")
    plt.show()

def graf_PCA(attributes, lables):
    fractions = []
    total_eig, _ = np.linalg.eigh(ML.covariance(attributes))
    total_eig = np.sum(total_eig)
    for i in range(attributes.shape[0]):
        if i == 0:
            continue
        _, reduces_attributes = ML.PCA(attributes, i)
        PCA_means, _  = np.linalg.eigh(ML.covariance(reduces_attributes))
        PCA_means = np.sum(PCA_means)
        fractions.append(PCA_means/total_eig)
    fractions.append(1)
    plt.plot(range(1,7), fractions, marker='o')
    plt.plot(range(1,7), [0.97]*6, '--')
    plt.grid(True)
    plt.ylabel('Fraction of the retained variance')
    plt.xlabel('Number of dimensions')
    plt.title("PCA variability analysis")
    plt.savefig(f"{os.getcwd()}/Image/PCA-Analysis.png")
    plt.show()

def graph_corr(attributes,labels):
    # Correlation of full dataset
    corr_attr = np.corrcoef(attributes)
    sns.heatmap(corr_attr, cmap='Greens')
    plt.title("Total attribute correlation")
    plt.savefig(f"{os.getcwd()}/Image/Dataset-correlation.png")
    plt.show()
    #correlation of males
    males = attributes[:, labels==0]
    corr_attr = np.corrcoef(males)
    sns.heatmap(corr_attr, cmap='Blues')
    plt.title(f"{class_label[0]} attribute correlation")
    plt.savefig(f"{os.getcwd()}/Image/{class_label[0]}-correlation.png")
    plt.show()
    #correlation of females
    females= attributes[:, labels==1]
    corr_attr = np.corrcoef(females)
    sns.heatmap(corr_attr, cmap="Reds")
    plt.savefig(f"{os.getcwd()}/Image/{class_label[1]}-correlation.png")
    plt.title(f"{class_label[1]} attribute correlation")
    plt.show()

if __name__ == "__main__":
    path = os.path.abspath("data/Train.txt")
    [attributes, labels] = load(path)
    for i in range(attributes.shape[0]):
        print(f"max {i}: {max(attributes[i])}", end="\t")
        print(f"min {i}: {min(attributes[i])}")
    print(f"Attribute dimensions: {attributes.shape[0]}")
    print(f"Points on the dataset: {attributes.shape[1]}")
    print(
        f"Distribution of labels (0, 1): {labels[labels==0].shape}, {labels[labels==1].shape}"
    )
    print(f"Possible classes: {class_label[0]}, {class_label[1]}")


    graficar(attributes)
    graf_LDA(attributes, labels)
    graf_PCA(attributes, labels)
    graph_corr(attributes, labels)

