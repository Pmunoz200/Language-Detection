import numpy as np
import pandas as pd
import os
from tabulate import tabulate
import matplotlib.pyplot as plt
from MLandPattern import MLandPattern as ML


def load(pathname, vizualization=0):
    df = pd.read_csv(pathname, header=None)
    if vizualization:
        print(df.head())
    attribute = np.array(df.iloc[:, 0 : len(df.columns) - 1])
    attribute = attribute.T
    label = np.array(df.iloc[:, -1])

    return attribute, label


def regression_train(attributes, labels, lamda, pi=[0.5], quadratic = 0, k = 5, pca = 0):
    minDCFvalues1=[]
    minDCFvalues2=[]
    minDCFvalues5=[]
    title = "Logarithmic regression" if not quadratic else "Quadratic regression"
    title = title + f"PCA: {pca}" if pca else title
    if quadratic==1:
        print(f"Quadratic regression with k-fold = {k}")
    if quadratic==0:
        print(f"Logarithmic regression with k-fold = {k}")
    for l in lamda:
        for p in pi:
            [_, _, accuracy, DCFnorm, minDCF] = ML.k_fold(
                k,
                attributes,
                labels,
                priorProb,
                "regression",
                l=l,
                pi=p,
                quadratic=quadratic,
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
    plt.semilogx(l_list, minDCFvalues1, label=f'π= 0.1')
    plt.semilogx(l_list, minDCFvalues2, label=f'π= 0.2')
    plt.semilogx(l_list, minDCFvalues5, label=f'π= avg(0.1, 0.5)')
    plt.xlabel('lambda')
    plt.ylabel('minDCF')
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ### Defining constants
    tableKFold = []
    pi=[0.1, 0.2, 0.5]
    l_list = np.logspace(-6, 6, 10, base=10)
    path = os.path.abspath("data/Train.txt")
    [full_train_att, full_train_label] = load(path)
    priorProb = ML.vcol(np.ones(2) * 0.5)
    q=[0,1]
    k=5
    
    standard_deviation = np.std(full_train_att)
    z_data = ML.center_data(full_train_att) / standard_deviation

    ## LOGISTIC REGRESSIONS ##
    regression_train(full_train_att, full_train_label, l_list, pi=pi, quadratic=0) # Full dataset
    regression_train(z_data, full_train_label, l_list, pi=pi, quadratic=0) # Z-norm dataset
    regression_train(full_train_att, full_train_label, l_list, pi=pi, pca=5, quadratic=0) # With PCA 5 applied


    ## QUADRATIC REGRESSIONS ##    
    regression_train(full_train_att, full_train_label, l_list,pi=pi, quadratic=1)
    regression_train(z_data, full_train_label,l_list, pi=pi, quadratic=1)
    regression_train(full_train_att, full_train_label, l_list, pi=pi, pca=5, quadratic=1)

    
      
    searchL=float(input("Enter a lambda value you want to search for: "))
    headers = [
    "Model",
    "π=0.1 DCF/minDCF",
    "π=0.2 DCF/minDCF",
    "π=0.5 DCF/minDCF",
    "π=0.1 Z-norm DCF/minDCF",
    "π=0.2 Z-norm DCF/minDCF",
    "π=0.5 Z-norm DCF/minDCF",
    ]
    pit = [0.1,0.2,0.5]
    cont = 0
    prev_1 = 0
    for i in q:
        for x in pit:
            if i==0:
                tableKFold.append([f"LR {x}"])
            if i==1:
                tableKFold.append([f"QLR {x}"]) 
            for p in pi:
                [_, _, accuracy, DCFnorm, minDCF] = ML.k_fold(
                        k,
                        full_train_att,
                        full_train_label,
                        priorProb,
                        "regression",
                        l=searchL,
                        pi=p,
                        quadratic=i,
                        pit=x
                    )
                if p == 0.1:
                    prev_1 = minDCF
                elif p == 0.5:
                    minDCF = (prev_1 + minDCF) / 2
                tableKFold[cont].append([DCFnorm, minDCF])
            for p in pi:
                [_, _, accuracy, DCFnorm, minDCF] = ML.k_fold(
                        k,
                        z_data,
                        full_train_label,
                        priorProb,
                        "regression",
                        l=searchL,
                        pi=p,
                        quadratic=i,
                        pit=x
                    )
                if p == 0.1:
                    prev_1 = minDCF
                elif p == 0.5:
                    minDCF = (prev_1 + minDCF) / 2
                tableKFold[cont].append([DCFnorm, minDCF])
            cont += 1
   
    print(tabulate(tableKFold, headers))
    # [_, _, accuracy, DCFnorm, minDCF] = ML.k_fold(
    #                     k,
    #                     full_train_att,
    #                     full_train_label,
    #                     priorProb,
    #                     "regression",
    #                     l=10**-3,
    #                     pi=0.5,
    #                     quadratic=0,
    #                     pit=0.5
    #                 )  
