import numpy as np
import pandas as pd
import os
from tabulate import tabulate
from matplotlib import pyplot as plt
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

def call_GMM(attributes, labels, prior_probability, models, k_value=5, m = 0, pi = [0.5]):
    tableGMM = []
    total_iter = len(models)*len(pi)
    perc = 0
    cont = 0
    headers = []
    headers.append([f"MinDCF for {pi} prior probability"])
    ### ------------- K-FOLD CROSS-VALIDATION AND PCA ---------------------- ####
    result_minDCF = []
    c2 = 1
    list_minDCF = []
    for model in models:
        tableGMM.append([model])
        ref = model.split(":")
        mod = ref[0]
        constrains = eval(ref[1])
        tableGMM[cont].append([])
        if not m:
            [S, _, _, minDCF] = ML.k_fold(
            k_value, attributes, labels, prior_probability, model=mod, niter=constrains[1],alpha=constrains[0], psi=constrains[2], pi=pi
            )
        else:
            [S, _, _, minDCF] = ML.k_fold(
                k_value, attributes, labels, prior_probability, model=mod, niter=constrains[1],alpha=constrains[0], psi=constrains[2], PCA_m=m, pi=pi
                )
            
        for p in pi:
            if p == 0.5:
                minDCF[p] = (minDCF[0.1] + minDCF[p]) / 2
                result_minDCF.append(minDCF[p])
            tableGMM[cont][c2].append([minDCF[p]])
            perc += 1
            list_minDCF.append(minDCF)
            print(f"{round(perc * 100 / total_iter, 2)}%", end=" ")
        cont += 1
    print("\tminDCF table")
    print(tabulate(tableGMM, headers=headers))
    return result_minDCF, S

def graph_data(raw_results, z_results, model, pca = 0, p_labels = 1):
    attribute = {
    "Raw": raw_results,
    "Z-Score": z_results
    }

    maxVal = max(max(attribute["Raw"]), max(attribute["Z-Score"]))

    x = np.arange(len(raw_results))  # the label locations
    print(x)
    width = 0.25  # the width of the bars
    multiplier = 0

    labels = np.power(2, x)

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in attribute.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        if p_labels:
            ax.bar_label(rects, padding=3) 
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('minDCF')
    ax.set_xticks(x + width/2, labels)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, maxVal + 0.1)
    if pca != 0:
        plt.savefig(f"{os.getcwd()}/Image/GMM-{model}-PCA.png")
    else:
        plt.savefig(f"{os.getcwd()}/Image/GMM-{model}.png")
    plt.show()


if __name__ == "__main__":
    
    ## Path for debbug ##
    path = "PythonCode/Data/Train.txt"
    ## Path for normal running ##
    path = "data/Train.txt"
    [full_train_att, full_train_label] = load(path)
    print(os.getcwd())
    priorProb = ML.vcol(np.ones(2) * 0.5)
    Cfn = 1
    Cfp = 1

    # The headers define which models the code will train. It is a list of all the models to try, and the format is:
    # [model: [alpha, G, psi]], where psi and alpha are hyperparameters, and G corresponds to the amount of clusters,
    # which are defined as 2^G.

    # headers = [
    # "GMM:[0.1, 2, 0.01]",
    # ]
    model = "GMM"
    labels = 1
    pca = 0
    headers = [
    f"{model}:[0.1, 0, 0.01]",
    f"{model}:[0.1, 1, 0.01]",
    f"{model}:[0.1, 2, 0.01]",
    f"{model}:[0.1, 3, 0.01]",
    f"{model}:[0.1, 4, 0.01]",
    f"{model}:[0.1, 5, 0.01]",
    ]
    # headers = ["GMM:[0.1, 2, 0.01]","Tied:[0.1, 5, 0.01]"]
    pi = [0.1, 0.2,  0.5]
    standard_deviation = np.std(full_train_att)
    z_data = ML.center_data(full_train_att) / standard_deviation
    
    print("Raw data")
    [raw_values, S_raw] = call_GMM(full_train_att, full_train_label, priorProb, pi=pi, models=headers, m=pca)
    print("Z-norm data")
    [z_values, S_z] = call_GMM(z_data, full_train_label, priorProb, pi=pi, models=headers, m=pca)
    headers = ["Tied:[0.1, 5, 0.01]"]
    pca = 5
    print("Raw data")
    [raw_values, S_raw] = call_GMM(full_train_att, full_train_label, priorProb, pi=pi, models=headers, m=pca)
    print("Z-norm data")
    [z_values, S_z] = call_GMM(z_data, full_train_label, priorProb, pi=pi, models=headers, m=pca)
    graph_data(raw_values, z_values, f"{model} with PCA" if pca else f"model", p_labels=labels)
    np.save("./Score_GMM", S_raw)

    # Bayes Error Plot
    pred_raw = ML.calculate_model(S_raw, full_train_att, "Generative", priorProb)
    conf_RAW = ML.ConfMat(pred_raw, full_train_label)
    ML.BayesErrorPlot(S_raw, full_train_label,conf_RAW,Cfn, Cfp)
    print("Z-norm data")
    [z_values, S_z] = call_GMM(z_data, full_train_label, priorProb, pi=pi, models=headers)


