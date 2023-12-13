import numpy as np
import pandas as pd
import os
from tabulate import tabulate
import sys
import matplotlib.pyplot as plt

from MLandPattern import MLandPattern as ML

def load(pathname, vizualization=0):
    df = pd.read_csv(pathname, header=None)
    if vizualization:
        print(df.head())
    attribute = np.array(df.iloc[:, 0: len(df.columns) - 1])
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

def svm_train(attributes, labels, C, model= "svm", pi=[0.5], k = 5, pca = 0):
    minDCFvalues1=[]
    minDCFvalues2=[]
    minDCFvalues5=[]
    title = "Support Vector Machine"
    title += f"with kernel model {model}." if model != "svm" else "."
    title = title + f"PCA: {pca}" if pca else title
    for c in C:
        for p in pi:
            [_, _, accuracy, DCFnorm, minDCF] = ML.k_fold(
                k,
                attributes,
                labels,
                priorProb,
                model=model,
                C=c,
                pi=p,
                PCA_m=pca
            )
            if p== 0.1:
                minDCFvalues1.append(minDCF)
                print(minDCFvalues1)
            if p== 0.2:
                minDCFvalues2.append(minDCF)
            if p== 0.5:
                print(minDCFvalues1[-1])
                val = (minDCF + minDCFvalues1[-1])/2
                minDCFvalues5.append(val)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(C, minDCFvalues1, label=f'π= 0.1')
    plt.semilogx(C, minDCFvalues2, label=f'π= 0.2')
    plt.semilogx(C, minDCFvalues5, label=f'π= avg(0.1, 0.5)')
    plt.xlabel('lambda')
    plt.ylabel('minDCF')
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ## Path for debbug ##
    # path = "PythonCode/Data/Train.txt"

    ## Path for normal running ##
    path = "Data/Train.txt"
    [full_train_att, full_train_label] = load(path)

    priorProb = ML.vcol(np.ones(2) * 0.5)

    pi = [0.1, 0.2, 0.5]
    Cfn = 1
    Cfp = 1
    initial_C = np.logspace(-6, 6, 10)
    initial_K = 1
    k = 5
    headers = [
    "Model",
    "π=0.5 DCF/minDCF",
    "π=0.3 DCF/minDCF"
    "π=0.7 DCF/minDCF",
    ]

    ### -------full dataset ----- ###
    svm_train(full_train_att, full_train_label, initial_C, pi=pi)
    svm_train(full_train_att, full_train_label, initial_C, pi=pi, model="polynomial")
    svm_train(full_train_att, full_train_label, initial_C, pi=pi, model="radial")


    ### -------z-score----------- ###
    standard_deviation = np.std(full_train_att)
    z_data = ML.center_data(full_train_att) / standard_deviation

    svm_train(z_data, full_train_label, initial_C, pi=pi)
    svm_train(z_data, full_train_label, initial_C, pi=pi, model="polynomial")
    svm_train(z_data, full_train_label, initial_C, pi=pi, model="radial")
    
    # ### --------z-score----------###
    # standard_deviation = np.std(full_train_att)
    # z_data = ML.center_data(full_train_att) / standard_deviation
    # section_size = int(full_train_att.shape[1] / k)
    # low = 0
    # all_values = np.c_[z_data.T, full_train_label]
    # all_values = np.random.permutation(all_values)
    # attributes = all_values[:, 0:12].T
    # labels = all_values[:, -1].astype("int32")
    # high = section_size
    # if high > attributes.shape[1]:
    #     high = attributes.shape
    # test_att = attributes[:, low:high]
    # test_labels = labels[low:high]
    # train_att = attributes[:, :low]
    # train_label = labels[:low]
    # train_att = np.hstack((train_att, attributes[:, high:]))
    # train_label = np.hstack((train_label, labels[high:]))

    # minDCFvalues5Z = []
    # minDCFvalues3Z = []
    # minDCFvalues7Z = []

    # for p in pi:
    #     for C in initial_C:
    #         contrain = C

    #         k = initial_K * np.power(10, 0)
    #         [SPost, Predictions, accuracy] = ML.svm(
    #             train_att, train_label, test_att, test_labels, contrain, K=k,
    #         )
    #         confusion_matrix = ML.ConfMat(Predictions, test_labels)
    #         DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, p, Cfn, Cfp)
    #         (minDCF, FPRlist, FNRlist, _) = ML.minCostBayes(
    #             SPost, test_labels, p, Cfn, Cfp
    #         )
    #         if p == 0.5:
    #             minDCFvalues5Z.append(minDCF)
    #         if p == 0.3:
    #             minDCFvalues3Z.append(minDCF)
    #         if p == 0.7:
    #             minDCFvalues7Z.append(minDCF)
    # tableFullZ=[]
    # cont=0
    # for p in pi:
    #     tableFullZ.append([f"SVM pi= {p}"])
    #     for x in pi:
    #         contrain = searchC
    #         k = initial_K * np.power(10, 0)
    #         [SPost, Predictions, accuracy] = ML.svm(
    #             train_att, train_label, test_att, test_labels, contrain, K=k,pit=p
    #         )
    #         confusion_matrix = ML.ConfMat(Predictions, test_labels)
    #         DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, x, Cfn, Cfp)
    #         (minDCF, FPRlist, FNRlist, _) = ML.minCostBayes(
    #             SPost, test_labels, x, Cfn, Cfp
    #         )
    #         tableFullZ[cont].append([DCFnorm,minDCF])
    #     cont+=1