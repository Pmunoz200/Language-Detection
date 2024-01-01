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
        [_, _, accuracy, minDCF] = ML.k_fold(
                k,
                attributes,
                labels,
                priorProb,
                model=model,
                C=c,
                pi=pi,
                PCA_m=pca
            )
        for p in pi:
            if p== 0.1:
                minDCFvalues1.append(minDCF[p])
                print(minDCFvalues1)
            if p== 0.2:
                minDCFvalues2.append(minDCF[p])
            if p== 0.5:
                print(minDCFvalues1[-1])
                val = (minDCF[p] + minDCFvalues1[-1])/2
                minDCFvalues5.append(val)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(C, minDCFvalues1, label=f'π= 0.1')
    plt.semilogx(C, minDCFvalues2, label=f'π= 0.2')
    plt.semilogx(C, minDCFvalues5, label=f'π= avg(0.1, 0.5)')
    plt.xlabel('Constant C')
    plt.ylabel('minDCF')
    plt.title(title)
    plt.legend()
    plt.show()

def svm_gamma_train(attributes, labels, C, model= "svm", pi=[0.1,0.2, 0.5], k = 5, pca = 0, gamma = 1):
    minDCFvalues1=[]
    minDCFvalues2=[]
    minDCFvalues5=[]
    title = "Support Vector Machine"
    title += f"with kernel model {model}." if model != "svm" else "."
    title = title + f"PCA: {pca}" if pca else title
    for g in gamma:
        [_, _, accuracy, minDCF] = ML.k_fold(
                k,
                attributes,
                labels,
                priorProb,
                model=model,
                C=C,
                pi=pi,
                PCA_m=pca,
                gamma=g
            )
        for p in pi:
            if p== 0.1:
                minDCFvalues1.append(minDCF[p])
                print(minDCFvalues1)
            if p== 0.2:
                minDCFvalues2.append(minDCF[p])
            if p== 0.5:
                print(minDCFvalues1[-1])
                val = (minDCF[p] + minDCFvalues1[-1])/2
                minDCFvalues5.append(val)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(gamma, minDCFvalues1, label=f'π= 0.1')
    plt.semilogx(gamma, minDCFvalues2, label=f'π= 0.2')
    plt.semilogx(gamma, minDCFvalues5, label=f'π= avg(0.1, 0.5)')
    plt.xlabel('gamma')
    plt.ylabel('minDCF')
    plt.title(title)
    plt.legend()
    plt.show()


def svm_b_c_train(attributes, labels, C, dim, c, model= "polynomial", pi=[0.1, 0.5], k = 5, pca = 0):
    minDCFvalues1=0
    graph_vals = []
    title = "Support Vector Machine"
    title += f"with kernel model {model}." if model != "svm" else "."
    title = title + f"PCA: {pca}" if pca else title
    lists = 0
    for constrain in C:
        cont = 0
        for b in dim:
            for const in c:
                if lists < 4:
                    graph_vals.append([])
                    lists += 1
                [_, _, accuracy, minDCF] = ML.k_fold(
                    k,
                    attributes,
                    labels,
                    priorProb,
                    model=model,
                    C=constrain,
                    pi=pi,
                    PCA_m=pca,
                    dim=b,
                    c=const
                )
                for p in pi:
                    if p== 0.1:
                        minDCFvalues1 = minDCF[p]
                    if p== 0.5:
                        print(minDCFvalues1)
                        val = (minDCF[p] + minDCFvalues1)/2
                        graph_vals[cont].append(val)
                        cont += 1
    
    plt.figure(figsize=(10, 6))
    cont = 0
    for b in range(len(dim)):
        for i in range(len(c)):
            plt.semilogx(C, graph_vals[cont], label=f'dim = {dim[b]} c= {c[i]}')
            cont += 1
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ## Path for debbug ##
    path = "PythonCode/Data/Train.txt"

    ## Path for normal running ##
    path = "Data/Train.txt"
    [full_train_att, full_train_label] = load(path)

    priorProb = ML.vcol(np.ones(2) * 0.5)

    pi = [0.1, 0.2, 0.5]
    Cfn = 1
    Cfp = 1
    initial_C = np.logspace(-6, 6, 10)
    gamma = np.logspace(-3,3,10)
    k = 5

    # ### -------full dataset ----- ###
    svm_train(full_train_att, full_train_label, initial_C, pi=pi)
    svm_train(full_train_att, full_train_label, initial_C, pi=pi, model="polynomial")
    svm_train(full_train_att, full_train_label, initial_C, pi=pi, model="radial")


    # ### -------z-score----------- ###
    standard_deviation = np.std(full_train_att)
    z_data = ML.center_data(full_train_att) / standard_deviation

    svm_train(z_data, full_train_label, initial_C, pi=pi)
    svm_train(z_data, full_train_label, initial_C, pi=pi, model="polynomial")
    svm_train(z_data, full_train_label, initial_C, pi=pi, model="radial")
    
    ### -------Gamma search----------- ###
    svm_gamma_train(full_train_att, full_train_label, 0.1, model="radial", gamma=gamma)
    svm_gamma_train(z_data, full_train_label, 0.1, model="radial", gamma=gamma)

    ## -------b and c search----------- ###
    full_dataset_C = np.logspace(-5, -1, 8)
    svm_b_c_train(full_train_att, full_train_label, full_dataset_C, [2, 3], [0,1], model="polynomial")
    z_dataset_C = np.logspace(-2, 1, 8)
    svm_b_c_train(z_data, full_train_label, z_dataset_C, [2, 3], [0,1], model="polynomial")



    ### -------Searching for the final table of values----------- ###
    c_vals = [0.1, 0.1, 0.0001]
    headers = [
        "Radial SVM",
        "Polynomial SVM z_norm",
        "Polynomial SVM",
    ]
    tableKFold = []
    pit = [0.1, 0.2, 0.5]
    gamma = 0.01
    cont = 0
    for x in range(len(headers)):
        tableKFold.append([f"SVM {x}"])
        if headers[x].split(" ")[-1] == "z_norm":
            [_, _, accuracy, minDCF] = ML.k_fold(
                    k,
                    z_data,
                    full_train_label,
                    priorProb,
                    headers[x].split(" ")[0].lower(),
                    pi=pi,
                    C=c_vals[x]
                )
        else:
            [_, _, accuracy, minDCF] = ML.k_fold(
                    k,
                    full_train_att,
                    full_train_label,
                    priorProb,
                    headers[x].split(" ")[0].lower(),
                    pi=pi,
                    C=c_vals[x],
                    gamma=gamma
                )
        for p in pi:
            if p == 0.5:
                print(minDCF)
                minDCF[p] = (minDCF[0.1] + minDCF[p]) / 2
                print(minDCF)
            tableKFold[cont].append([minDCF[p]])
        cont += 1
   
    print(tabulate(tableKFold, headers))