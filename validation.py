import numpy as np
import pandas as pd
import scipy
import os
from tabulate import tabulate
import sys
import matplotlib.pyplot as plt
from MLandPattern import MLandPattern as ML

tableTrain = []
tableTest = []
headers = [
    "Gaussian MVG PCA 8  kfold=20",
    "GMM Full Full, [0.1, 2, 0.01]",
    "LR PCA8 LDA3 kfold=20 l=10-6",
    "QR PCA7 k_fold=20 l=10-1",
]
pi = 0.5
Cfn = 1
Cfp = 1


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

def calculate_logreg(test_points, w, b, quadratic = 0):
    xt_2 = np.dot(test_points.T, test_points).diagonal().reshape((1, -1)) if quadratic else np.array([])
    test_points = np.vstack((xt_2, test_points))
    S = np.dot(w.T, test_points) + b
    return S

def calculate_svm(x, zi, train_att, test_feat, args):
    S = x * zi
    if args["model"]=="polynomial":
        S = np.dot(S, ML.polynomial_kernel(train_att, test_feat, args["dim"], args["c"], args["epsilon"]))
    elif args["model"]=="radial":
        S = np.dot(S, ML.radial_kernel(train_att, test_feat, args["gamma"], args["epsilon"]))
    return S

def calculate_gmm(test_attributes, test_labels, class_mu, class_c, class_w):
    densities = []
    for i in np.unique(test_labels):
        ll = np.array(ML.ll_gaussian(test_attributes, class_mu[i], class_c[i]))
        Sjoin = ll + np.log(class_w[i].reshape((-1, 1)))
        logdens = scipy.special.logsumexp(Sjoin, axis=0)
        densities.append(logdens)
    return np.array(densities)

def calculate_model(
    args,
    test_points,
    model,
    prior_probability,
    test_labels,
    train_att=[],
    train_labels=[],
    pi = [0.5]
):
    model = model.lower()
    minDCF_dic = {}
    funct = lambda s: 1 if s > 0 else 0
    if model == "gaussian":
        multi_mu = args[0]
        cov = args[1]
        densities = []
        for i in np.unique(test_labels):
            densities.append(
                ML.logLikelihood(test_points, ML.vcol(multi_mu[i, :]), cov)
            )
        S = np.array(densities)
        logSJoint = S + np.log(prior_probability)
        logSMarginal = ML.vrow(scipy.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal
        SPost = np.exp(logSPost)
        predictions = np.argmax(SPost, axis=0)
    elif model == "regression" or model == "quadratic":
        xt_2 = np.dot(test_points.T, test_points).diagonal().reshape((1, -1)) if model == "quadratic" else np.array([])
        test_points = np.vstack((xt_2, test_points))
        w = args[0]
        b = args[1]
        S = np.dot(w.T, test_points) + b
        predictions = np.array(list(map(funct, S)))
    elif model == "gmm":
        class_mu = args[0]
        class_c = args[1]
        class_w = args[2]
        densities = []
        for i in np.unique(test_labels):
            ll = np.array(ML.ll_gaussian(test_points, class_mu[i], class_c[i]))
            Sjoin = ll + np.log(class_w[i].reshape((-1, 1)))
            logdens = scipy.special.logsumexp(Sjoin, axis=0)
            densities.append(logdens)
        S = np.array(densities)
        predictions = np.argmax(S, axis=0)
    elif model == "svm":
        x = args[0]
        zi = 2 * train_labels - 1
        S = x * zi
        S = np.dot(S, ML.polynomial_kernel(train_att, test_points, 2, 1 , 0.1))
        predictions = np.where(S > 0, 1, 0)
    for p in pi:
        (minDCF, _, _, _) = ML.minCostBayes(S, test_labels, pi, Cfn, Cfp)
        minDCF_dic[p] = minDCF
    ML.BayesErrorPlot(S, test_labels, Cfn, Cfp, model)
    error = np.abs(test_labels - predictions)
    error = np.sum(error) / test_labels.shape[0]
    return predictions, (1 - error),  minDCF_dic


if __name__ == "__main__":

    minDCffull_list = []

    ## Hyper-parameters Quadratic Regression
    l = 0.1
    pit = 0.5
    ## Hyper-parameters SVM
    C = 0.1
    dim = 2
    c = 1
    gamma = 0.01

    ## Hyper-parameters GMM
    niter = 4
    alpha = 0.1
    psi = 0.01

    ## Hyper-parameters k-fold
    k = 5
    pi = [0.1, 0.5]
    pathTrain = os.path.abspath("Data/Train.txt")
    [full_train_att, full_train_label] = load(pathTrain)

    pathTest = os.path.abspath("Data/Test.txt")
    [full_test_att, full_test_labels] = load(pathTest)

    priorProb = ML.vcol(np.ones(2) * 0.5)
    tableTest.append([])
    tableTrain.append([])

    # ## Quadratic Regression
    # [_, _, minDCF, w, b] = ML.k_fold(
    #     k,
    #     full_train_att,
    #     full_train_label,
    #     priorProb,
    #     "regression",
    #     pi=pi,
    #     pit=pit,
    #     quadratic=1,
    #     l=l,
    #     final=1,
    # )

    # ## Test Quadratic Regression
    # scr_quadra = calculate_logreg(full_test_att, w, b, 1)
    # for p in pi:
    #     (minDCF, _, _, _) = ML.minCostBayes(scr_quadra, full_test_labels, p, Cfn, Cfp)
    #     minDCffull_list.append(minDCF)
    # print("Quadratic Regression MinDCF: ", minDCffull_list, "Mean: ", np.mean(minDCffull_list))

    # ML.BayesErrorPlot(scr_quadra, full_test_labels, Cfn, Cfp, "Quadratic Regression")

    # ## Calculate Radial SVM
    # [_,  minDCF, attribute_tra_rbm, labels_tra_rbm, x] = ML.k_fold(
    #     k,
    #     full_train_att,
    #     full_train_label,
    #     priorProb,
    #     "radial",
    #     pi=[0.1, 0.5],
    #     C=C,
    #     gamma=gamma,
    #     final=1,
    # )

    # ## Test Radial SVM
    # zi = 2 * labels_tra_rbm - 1
    # scr_poly = calculate_svm(x, zi, attribute_tra_rbm, full_test_att, {"model":"radial", "gamma":gamma, "epsilon":0.1})

    # for p in pi:
    #     (minDCF, _, _, _) = ML.minCostBayes(scr_poly, full_test_labels, p, Cfn, Cfp)
    #     minDCffull_list.append(minDCF)
    # print("Radial SVM MinDCF: ", minDCffull_list[-2:], "Mean: ", np.mean(minDCffull_list[-2:]))
    # ML.BayesErrorPlot(scr_poly, full_test_labels, Cfn, Cfp, "Radial SVM")

    # Calculate Polynomial SVM
    [_,  minDCF, attribute_tra_poly, labels_tra_poly, x] = ML.k_fold(
        k,
        full_train_att,
        full_train_label,
        priorProb,
        "polynomial",
        pi=[0.1, 0.5],
        C=C,
        dim=dim,
        c=c,
        final=1,
    )

    ## Test Polynomial SVM
    zi = 2 * labels_tra_poly - 1
    scr_poly = calculate_svm(x, zi, attribute_tra_poly, full_test_att, {"model":"polynomial", "dim":dim, "c":c, "epsilon":0.1})
    print("Polynomial SVM MinDCF: ", minDCffull_list[-2:], "Mean: ", np.mean(minDCffull_list[-2:]))
    ML.BayesErrorPlot(scr_poly, full_test_labels, Cfn, Cfp, "Polynomial SVM")

    ## Calculate GMM
    [_, _, minDCF, multi_mu, cov, w] = ML.k_fold(
        k,
        full_train_att,
        full_train_label,
        priorProb,
        "GMM",
        pi=pi,
        alpha=alpha,
        psi=psi,
        niter=niter,
        final=1,
    )

    ## Test GMM
    scr_gmm = calculate_gmm(full_test_att, full_test_labels, multi_mu, cov, w)
    for p in pi:
        (minDCF, _, _, _) = ML.minCostBayes(scr_gmm, full_test_labels, p, Cfn, Cfp)
        minDCffull_list.append(minDCF)
    print("GMM MinDCF: ", minDCffull_list[-2:], "Mean: ", np.mean(minDCffull_list[-2:]))
    ML.BayesErrorPlot(scr_gmm, full_test_labels, Cfn, Cfp, "GMM")