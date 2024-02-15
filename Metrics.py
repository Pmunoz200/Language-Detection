import numpy as np
from tabulate import tabulate
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b

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


def ConfMat(predicted, actual):
    labels = np.unique(np.concatenate((actual, predicted)))

    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for true_label, predicted_label in zip(actual, predicted):
        true_index = np.where(labels == true_label)[0]
        predicted_index = np.where(labels == predicted_label)[0]
        matrix[predicted_index, true_index] += 1

    return matrix


def OptimalBayes(llr, labels, pi, Cfn, Cfp, model):
    if llr.ndim > 1:
        llr = (Cfn * llr[0, :]) / (Cfp * llr[1, :])
    log_odds = llr
    threshold = -np.log((pi * Cfn) / ((1 - pi) * Cfp))
    decisions = np.where(log_odds > threshold, 1, 0)

    tp = np.sum(np.logical_and(decisions == 1, labels == 1))
    fp = np.sum(np.logical_and(decisions == 1, labels == 0))
    tn = np.sum(np.logical_and(decisions == 0, labels == 0))
    fn = np.sum(np.logical_and(decisions == 0, labels == 1))

    confusion_matrix = np.array([[tn, fn], [fp, tp]])

    return confusion_matrix


def Bayes_risk(confusion_matrix, pi, Cfn, Cfp):
    M01 = confusion_matrix[0][1]
    M11 = confusion_matrix[1][1]
    M10 = confusion_matrix[1][0]
    M00 = confusion_matrix[0][0]

    FNR = M01 / (M01 + M11)
    FPR = M10 / (M00 + M10)

    DCF = (pi * Cfn * FNR) + ((1 - pi) * Cfp * FPR)

    B = min(pi * Cfn, (1 - pi) * Cfp)

    DCFnorm = DCF / B

    return DCF, DCFnorm


def minCostBayes(llr, labels, pi, Cfn, Cfp):
    if llr.ndim > 1:
        llr = (Cfn * llr[0, :]) / (Cfp * llr[1, :])
    sortedLLR = np.sort(llr)
    # sortedLLR = pi * sortedLLR[0, :] / ((1 - pi) * sortedLLR[1, :])
    t = np.array([-np.inf, np.inf])
    t = np.append(t, sortedLLR)
    t = np.sort(t)
    DCFnorm = []
    FNRlist = []
    FPRlist = []
    for i in t:
        threshold = i
        funct = lambda s: 1 if s > i else 0
        decisions = np.array(list(map(funct, llr)))
        # decisions = np.where(llr > threshold, 1, 0)

        tp = np.sum(np.logical_and(decisions == 1, labels == 1))
        fp = np.sum(np.logical_and(decisions == 1, labels == 0))
        tn = np.sum(np.logical_and(decisions == 0, labels == 0))
        fn = np.sum(np.logical_and(decisions == 0, labels == 1))

        confusion_matrix = np.array([[tn, fn], [fp, tp]])
        M01 = confusion_matrix[0][1]
        M11 = confusion_matrix[1][1]
        M10 = confusion_matrix[1][0]
        M00 = confusion_matrix[0][0]

        FNR = M01 / (M01 + M11)
        FPR = M10 / (M00 + M10)

        [DCF, DCFnormal] = Bayes_risk(confusion_matrix, pi, Cfn, Cfp)

        FNRlist = np.append(FNRlist, FNR)
        FPRlist = np.append(FPRlist, FPR)
        DCFnorm = np.append(DCFnorm, DCFnormal)
    minDCF = min(DCFnorm)

    return minDCF, FPRlist, FNRlist



def BayesErrorPlot(llr, labels, confusion_matrix, Cfn, Cfp):
    DCFlist = []
    minDCFlist = []
    effPriorLogOdds = np.linspace(-6, 6, 15)
    prior = 1 / (1 + np.exp(-effPriorLogOdds))

    for i in prior:
        (_, DCFnorm) = Bayes_risk(confusion_matrix, i, Cfn, Cfp)
        (minDCF, _, _) = minCostBayes(llr, labels, i, Cfn, Cfp)
        DCFlist = np.append(DCFlist, DCFnorm)
        minDCFlist = np.append(minDCFlist, minDCF)

    plt.figure()
    plt.plot(effPriorLogOdds, DCFlist, label="DCF", color="r")
    plt.plot(effPriorLogOdds, minDCFlist, label="min DCF", color="b", linestyle="--")
    plt.ylim([0, 3])
    plt.xlim([-6, 6])
    plt.show()

def logreg_obj(v, DTR, LTR, l, n):
    # compute J(w,b) = l/2*||w||^2 + 1/n*sum(log(1+e^(-zi*(wt*xi+b))))
    # given zi = 1 if ci=1 and =0 if ci=0
    
    w, b = v[0:-1], v[-1]
    J = l/2 * np.dot(w, w.T)                 #numpy.linalg.norm(w)
    for i in range(n):
        zi = 1 if LTR[i]==1 else -1
        if DTR.ndim > 1:
            J += np.logaddexp(0, -zi * (np.dot(w.T, DTR[:,i]) + b)) / n
        else:
            J += np.logaddexp(0, -zi * (np.dot(w.T, DTR[i]) + b)) / n

    return J

def linear_regression_2C(feat, lab, l):
    x0 = np.zeros(feat.shape[0] + 1)
    
    opt,_1,_2 = fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, args=(feat, lab, l, lab.shape[0]))
    wopt, bopt = opt[0:-1], opt[-1]
    
    return wopt, bopt

def score_calibration(K, DTR, LTR, l, LOO=False):  
    """K is n of folds, lambda and model must be a function that returns an array of predictions"""
    #given DTR, LTR and K compute K-folds cross validation (Leave One Out with K = N)
    #give a model function that accepts DTR, LTR, DTV and gives out some predictions to be confronted with LTV
    if DTR.ndim == 1:
        DTR = ML.vrow(DTR)
    N = DTR.shape[1]
    if(LOO):
        K = N
    seglen = N//K
    #costs = []
    alls = []
    for i in range(K):
        mask = [True] * N
        mask[i*seglen : (i+1)*seglen] = [False] * seglen
        notmask = [not x for x in mask]
        DTRr = DTR[:,mask]
        LTRr = LTR[mask]
        DTV = DTR[:,notmask]
        #LTV = LTR[notmask]
        #LTV = LTR[notmask]
        wopt, bopt = linear_regression_2C(DTRr, LTRr, l)        
        #predictions = (numpy.dot(vrow(wopt), DTV) + bopt) > 0  
        scores = np.dot(ML.vrow(wopt), DTV) + bopt
        #cost = minimum_detection_cost(0.5, 1, 1, scores.ravel(), LTV)
        alls += scores.ravel().tolist()
        # accuracy = (predictions == LTV).sum()/LTV.__len__()
        #costs.append(cost)
    # print("minDCF: ", minimum_detection_cost(0.5, 1, 1, alls, LTR))
    return alls

if __name__ == "__main__":


    # ## Path for debbug ##
    # path = os.path.abspath("PythonCode/Data/Train.txt")

    ## Path for normal running ##
    path = os.path.abspath("data/Train.txt")
    [full_train_att, full_train_label] = load(path)

    priorProb = ML.vcol(np.ones(2) * 0.5)
    Cfn = 1
    Cfp = 1


    # Define the parameters for the models
    k = 5
    pi = [0.1, 0.5]
    standard_deviation = np.std(full_train_att)
    z_data = ML.center_data(full_train_att) / standard_deviation

    [SQuad, labels_quad, minDCF_quad] = ML.k_fold(k, full_train_att, full_train_label, priorProb, model="regression", pi=pi, pit=0.5, quadratic=1, score_calibration_flag=1)
    print("Quadratic MinDCF: ", (minDCF_quad[0.5] + minDCF_quad[0.1])/2)
    [SRadial, labels_radial, minDCF_radial] = ML.k_fold(k, full_train_att, full_train_label, priorProb, model="radial", pi=pi, pit=0.5, gamma=0.01, score_calibration_flag=1)
    print("MinDCF radial: ", (minDCF_radial[0.5] + minDCF_radial[0.1])/2)
    [Spoly, labels, minDCF] = ML.k_fold(k, z_data, full_train_label, priorProb, model="polynomial", pi=pi, pit=0.5, score_calibration_flag=1, C=0.1)
    print("Polynomial MinDCF: ", (minDCF[0.5] + minDCF[0.1])/2)
    [SGMM, labels_gmm, minDCF_gmm] = ML.k_fold(k, z_data, full_train_label, priorProb, model="GMM", pi=pi, pit=0.5, niter=2, alpha=0.1, psi=0.01, score_calibration_flag=1)
    print("GMM MinDCF: ", (minDCF_gmm[0.5] + minDCF_gmm[0.1])/2)

    ML.BayesErrorPlot(SQuad, labels_quad, Cfn, Cfp, model="quadratic")
    ML.BayesErrorPlot(SRadial, labels_radial, Cfn, Cfp, model="Radial")
    ML.BayesErrorPlot(Spoly, labels, Cfn, Cfp, model="polynomial")
    ML.BayesErrorPlot(SGMM, labels_gmm, Cfn, Cfp, model="GMM")

    # Calibrate the score
    pi = [0.1, 0.5]
    labels_quad = np.array(labels_quad)
    labels_radial = np.array(labels_radial)
    labels = np.array(labels)
    labels_gmm = np.array(labels_gmm)

    scQuad_cal = score_calibration(5, SQuad, labels_quad, 0.00001)
    scQuad_cal = np.array(scQuad_cal)
    labels_quad = labels_quad[:-1]
    print("Quadratic MinDCF Calibrated: ", (ML.minCostBayes(scQuad_cal, labels_quad, pi[1], Cfn, Cfp)[0] + ML.minCostBayes(scQuad_cal, labels_quad, pi[0], Cfn, Cfp)[0])/2)
    scRadial_cal = score_calibration(5, SRadial, labels_radial, 0.00001)
    scRadial_cal = np.array(scRadial_cal)
    labels_radial = labels_radial[:-1]
    print("MinDCF Naive Calibrated: ", (ML.minCostBayes(scRadial_cal, labels_radial, pi[1], Cfn, Cfp)[0] + ML.minCostBayes(scRadial_cal, labels_radial, pi[0], Cfn, Cfp)[0])/2)
    scPoly_cal = score_calibration(5, Spoly, labels, 0.00001)
    scPoly_cal = np.array(scPoly_cal)
    labels = labels[:-1]
    print("Polynomial MinDCF Calibrated: ", (ML.minCostBayes(scPoly_cal, labels, pi[1], Cfn, Cfp)[0] + ML.minCostBayes(scPoly_cal, labels, pi[0], Cfn, Cfp)[0])/2)
    scGMM_cal = score_calibration(5, SGMM, labels_gmm, 0.00001)
    scGMM_cal = np.array(scGMM_cal)
    labels_gmm = labels_gmm[:-1]
    print("GMM MinDCF Calibrated: ", (ML.minCostBayes(scGMM_cal, labels_gmm, pi[1], Cfn, Cfp)[0] + ML.minCostBayes(scGMM_cal, labels_gmm, pi[0], Cfn, Cfp)[0])/2)

    ML.BayesErrorPlot(scQuad_cal, labels_quad, Cfn, Cfp, model="quadratic Calibrated")
    ML.BayesErrorPlot(scRadial_cal, labels_radial, Cfn, Cfp, model="Radial Calibrated")
    ML.BayesErrorPlot(scPoly_cal, labels, Cfn, Cfp, model="polynomial Calibrated")
    ML.BayesErrorPlot(scGMM_cal, labels_gmm, Cfn, Cfp, model="GMM Calibrated")
