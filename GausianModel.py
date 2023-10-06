import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from tabulate import tabulate
import sys
from MLandPattern import MLandPattern as ML



def load(pathname, vizualization=0):
    df = pd.read_csv(pathname, header=None)
    if vizualization:
        print(df.head())
    attribute = np.array(df.iloc[:, 0 : len(df.columns) - 1])
    attribute = attribute.T
    # print(attribute)
    label = np.array(df.iloc[:, -1])

    return attribute, label


def split_db(D, L, fraction, seed=0):
    nTrain = int(D.shape[1] * fraction)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)


def gaussian_train(attributes, labels, headers, priorProb = 0, pi=[0.5], Cfn=1, Cfp=1):
    ### Parameter definition ###
    tableKFold = []
    priorProb = ML.vcol(np.ones(2) * 0.5) if not priorProb else priorProb
    ####
    tableKFold.append(["Full"])
    k_fold_value = 5
    c2 = 1
    list_minDCF = []
    list_DCF = []
    for model in headers:
        tableKFold[0].append([])
        for p in pi:
            [SPost, Predictions, accuracy, DCFnorm, minDCF] = ML.k_fold(
                k_fold_value, attributes, labels, priorProb, model=model, pi=p
            )
            tableKFold[0][c2].append(minDCF)
            list_minDCF.append(minDCF)
            list_DCF.append(DCFnorm)
        c2 += 1

    cont = 1
    for i in [7, 6]:
        tableKFold.append([f"PCA {i}"])
        c2 = 1
        for model in headers:
            tableKFold[cont].append([])
            for p in pi:
                [_, _, _, DCFnorm, minDCF] = ML.k_fold(
                    k_fold_value,
                    attributes,
                    labels,
                    priorProb,
                    model=model,
                    PCA_m=i,
                )
                tableKFold[cont][c2].append(minDCF)
                list_minDCF.append(minDCF)
                list_DCF.append(DCFnorm)
            c2 += 1
        cont += 1

    newHeaders = []
    print("PCA with k-fold")
    for i in headers:
        newHeaders.append(i + " MinDCF" + " ".join(str(p) for p in pi))
    print(tabulate(tableKFold, headers=newHeaders))


if __name__ == "__main__":
    ## For Debugging mode
    # path = os.path.abspath("PythonCode/data/Train.txt")

    ## For normal running
    path = os.path.abspath("data/Train.txt")

    [full_train_att, full_train_label] = load(path)


    standard_deviation = np.std(full_train_att)
    z_data = ML.center_data(full_train_att) / standard_deviation
    models = ["MVG", "Naive", "Tied Gaussian", "Tied Naive"]
    pi = [0.1, 0.5]
    print("Full dataset")
    gaussian_train(full_train_att, full_train_label,models, pi=pi)
    print("Z-Norm dataset")
    gaussian_train(z_data, full_train_label, models,pi=pi)
