import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import os
from tabulate import tabulate
import math
from matplotlib import pyplot as plt
from scipy.optimize import fmin_l_bfgs_b

class_label = ["Non-target Language", "Target Language"]

def load(pathname, vizualization=0):
    df = pd.read_csv(pathname, header=None)
    if vizualization:
        print(df.head())
    attribute = np.array(df.iloc[:, 0 : len(df.columns) - 1])
    attribute = attribute.T
    # print(attribute)
    label = np.array(df.iloc[:, -1])

    return attribute, label

def vcol(vector):
    """
    Reshape a vector row vector into a column vector
    :param vector: a numpy row vector
    :return: the vector reshaped as a column vector
    """
    column_vector = vector.reshape((vector.size, 1))
    return column_vector


def vrow(vector):
    """
    Reshape a vector column vector into a row vector
    :param vector: a numpy column vector
    :return: the vector reshaped as a row vector
    """
    row_vector = vector.reshape((1, vector.size))
    return row_vector

def mean_of_matrix_rows(matrix):
    """
    Calculates the mean of the rows of a matrix
    :param matrix: a matrix of numpy arrays
    :return: a numpy column vector with the mean of each row
    """
    mu = matrix.mean(1)
    mu_col = vcol(mu)
    return mu_col


def center_data(matrix):
    """
    Normalizes the data on the dataset by subtracting the mean
    to each element.
    :param matrix: a matrix of numpy arrays
    :return: a matrix of the input elements minus the mean for
    each row
    """
    mean = mean_of_matrix_rows(matrix)
    centered_data = matrix - mean
    return centered_data

def eigen(matrix):
    """
    Calculates the eigen value and vectors for a matrix
    :param matrix: Matrix of data points
    :return: eigen values, eigen vectors
    """
    if matrix.shape[0] == matrix.shape[1]:
        s, U = np.linalg.eigh(matrix)
        return s, U
    else:
        s, U = np.linalg.eig(matrix)
        return s, U

def covariance(matrix, centered=0):
    """
    Calculates the Sample Covariance Matrix of a centered-matrix
    :param matrix: Matrix of data points
    :param centered: Flag to determine if matrix data is centered (default is False)
    :return: The data covariance matrix
    """
    if not centered:
        matrix = center_data(matrix)
    n = matrix.shape[1]
    cov = np.dot(matrix, matrix.T)
    cov = np.multiply(cov, 1 / n)
    return cov

def covariance_within_class(matrix_values, label):
    """
    Calculates the average covariance within all the classes in a dataset
    :param matrix_values: matrix with the values associated to the parameters of the dataset
    :param label: vector with the label values associated with the dataset
    :return: a matrix with the total average covariance within each class
    """
    class_labels = np.unique(label)
    within_cov = np.zeros((matrix_values.shape[0], matrix_values.shape[0]))
    n = matrix_values.size
    for i in class_labels:
        centered_matrix = center_data(matrix_values[:, label == i])
        cov_matrix = covariance(centered_matrix, 1)
        cov_matrix = np.multiply(cov_matrix, centered_matrix.size)
        within_cov = np.add(within_cov, cov_matrix)
    within_cov = np.divide(within_cov, n)
    return within_cov


def covariance_between_class(matrix_values, label):
    """
    Calculates the total covariance between all the classes in a dataset
    :param matrix_values: matrix with the values associated to the parameters of the dataset
    :param label: vector with the label values associated with the dataset
    :return: a matrix with the covariance between each class
    """
    class_labels = np.unique(label)
    between_cov = np.zeros((matrix_values.shape[0], matrix_values.shape[0]))
    N = matrix_values.size
    m_general = mean_of_matrix_rows(matrix_values)
    for i in range(len(class_labels)):
        values = matrix_values[:, label == i]
        nc = values.size
        m_class = mean_of_matrix_rows(values)
        norm_means = np.subtract(m_class, m_general)
        matr = np.multiply(nc, np.dot(norm_means, norm_means.T))
        between_cov = np.add(between_cov, matr)
    between_cov = np.divide(between_cov, N)
    return between_cov


def between_within_covariance(matrix_values, label):
    """
    Calculates both the average within covariance, and the between covariance of all classes on a dataset
    :param matrix_values: matrix with the values associated to the parameters of the dataset
    :param label: vector with the label values associated with the dataset
    :return:a matrix with the total average covariance within each class, and the covariance between each class
    """
    Sw = covariance_within_class(matrix_values, label)
    Sb = covariance_between_class(matrix_values, label)
    return Sw, Sb

def PCA(attribute_matrix, m):
    """
    Calculates the PCA dimension reduction of a matrix to a m-dimension sub-space
    :param attribute_matrix: matrix with the datapoints, with each row being a point
    `param m` number of dimensions of the targeted sub-space
    :return: The matrix P defined to do the PCA approximation
    :return: The dataset after the dimensionality reduction
    """
    DC = center_data(attribute_matrix)
    C = covariance(DC, 1)
    s, U = eigen(C)
    P = U[:, ::-1][:, 0:m]
    return P, np.dot(P.T, attribute_matrix)

def LDA1(matrix_values, label, m):
    """
    Calculates the Lineal Discriminant Analysis to perform dimension reduction
    :param matrix_values: matrix with the datapoints, with each row being a point
    :param label: vector with the label values associated with the dataset
    :param m: number of dimensions of the targeted sub-space
    :return: the LDA directions matrix (W), and the orthogonal sub-space of the directions (U)
    """
    class_labels = np.unique(label)
    [Sw, Sb] = between_within_covariance(matrix_values, label)
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
    UW, _, _ = np.linalg.svd(W)
    U = UW[:, 0:m]
    return W, U

def multiclass_mean(matrix, labels):
    """
    Calculates the mean for each class in  dataset
    :param matrix: matrix of the datapoints of a dataset
    :param labels: row vector with the labels associated with each row of data points
    :return: A np matrix with size (# of classes, n) related with the mean associated with each class, in each dimension
    """
    class_labels = np.unique(labels)
    multi_mu = np.zeros((class_labels.size, matrix.shape[0]))
    n = matrix.size
    for i in class_labels:
        mu = mean_of_matrix_rows(matrix[:, labels == i])
        multi_mu[i, :] = mu[:, 0]
    return multi_mu

def multiclass_covariance(matrix, labels):
    """
    Calculates the Covariance for each class in  dataset
    :param matrix: matrix of the datapoints of a dataset
    :param labels: row vector with the labels associated with each row of data points
    :return: A np matrix with size (# of classes, n, n) related with the covariance associated with each class, in each dimension
    """
    class_labels = np.unique(labels)
    within_cov = np.zeros((class_labels.size, matrix.shape[0], matrix.shape[0]))
    n = matrix.size
    for i in class_labels:
        centered_matrix = center_data(matrix[:, labels == i])
        within_cov[i, :, :] = covariance(centered_matrix)
    return within_cov

def logLikelihood(X, mu, c, tot=0):
    """
    Calculates the Logarithmic Maximum Likelihood estimator
    :param X: matrix of the datapoints of a dataset, with a size (n x m)
    :param mu: row vector with the mean associated to each dimension
    :param c: Covariance matrix
    :param tot: flag to define if it returns value per datapoint, or total sum of logLikelihood (default is False)
    :return: the logarithm of the likelihood of the datapoints, and the associated gaussian density
    """
    logN = logpdf_GAU_ND(X, mu, c)
    if tot:
        return logN.sum()
    else:
        return logN

def logpdf_GAU_ND(x, mu, C):
    """
    Calculates the Logarithmic MultiVariate Gaussian Density for a set of vector values
    :param x: matrix of the datapoints of a dataset, with a size (n x m)
    :param mu: row vector with the mean associated to each dimension
    :param C: Covariance matrix
    :return: a matrix with the Gaussian Density associated with each point of X, over each dimension
    """
    M = C.shape[1]
    inv_C = np.linalg.inv(C)
    # print(inv_C.shape)
    [_, log_C] = np.linalg.slogdet(C)

    # print(log_C)
    log_2pi = -M * math.log(2 * math.pi)
    x_norm = x - mu
    inter_value = np.dot(x_norm.T, inv_C)
    dot_mul = np.dot(inter_value, x_norm)
    dot_mul = np.diag(dot_mul)

    y = (log_2pi - log_C - dot_mul) / 2
    return y

def ll_gaussian(x, mu, C):
    M = mu[0].shape[0]
    s = []
    for i in range(len(C)):
        inv_C = np.linalg.inv(C[i])
        # print(inv_C.shape)
        [_, log_C] = np.linalg.slogdet(C[i])
        log_2pi = -M * math.log(2 * math.pi)
        x_norm = x - vcol(mu[i])
        inter_value = np.dot(x_norm.T, inv_C)
        dot_mul = np.dot(inter_value, x_norm)
        dot_mul = np.diag(dot_mul)
        y = (log_2pi - log_C - dot_mul) / 2
        s.append(y)
    return s

def MVG_log_classifier(
    train_data, train_labels, test_data, prior_probability, test_label=[], final=0
):
    """
    Calculates the model of the MultiVariate Gaussian classifier on the logarithm dimension for a set of data, and applyes it to a test dataset
    :param train_date: matrix of the datapoints of a dataset used to train the model
    :param train_labels: row vector with the labels associated with each row of the training dataset
    :param test_data: matrix of the datapoints of a dataset used to test the model
    :param test_labels: row vector with the labels associated with each row of the test dataset
    :param prior_probability: col vector associated with the prior probability for each dimension
    :return S: matrix associated with the probability array
    :return predictions: Vector associated with the prediction of the class for each test data point
    :return acc: Accuracy of the validation set
    """
    class_labels = np.unique(train_labels)
    cov = multiclass_covariance(train_data, train_labels)
    # print(cov[0])
    multi_mu = multiclass_mean(train_data, train_labels)
    densities = []
    for i in range(class_labels.size):
        densities.append(logLikelihood(test_data, vcol(multi_mu[i, :]), cov[i]))
    S = np.array(densities)
    logSJoint = S + np.log(prior_probability)
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    predictions = np.argmax(logSPost, axis=0)

    if len(test_label) != 0:
        acc = 0
        for i in range(len(test_label)):
            if predictions[i] == test_label[i]:
                acc += 1
        acc /= len(test_label)
        acc = round(acc * 100, 2)
    if final:
        return logSPost, predictions, acc, multi_mu, cov
    else:
        return logSPost, predictions, acc

def Naive_log_classifier(
    train_data, train_labels, test_data, prior_probability, test_label=[]
):
    """
    Calculates the model of the Naive classifier on the logarithm realm for a set of data, and applyes it to a test dataset
    :param train_date: matrix of the datapoints of a dataset used to train the model
    :param train_labels: row vector with the labels associated with each row of the training dataset
    :param test_data: matrix of the datapoints of a dataset used to test the model
    :param test_labels: row vector with the labels associated with each row of the test dataset
    :param prior_probability: col vector associated with the prior probability for each dimension
    :return S: matrix associated with the probability array
    :return predictions: Vector associated with the prediction of the class for each test data point
    :return acc: Accuracy of the validation set
    """
    class_labels = np.unique(train_labels)
    cov = multiclass_covariance(train_data, train_labels)
    identity = np.eye(cov.shape[1])
    cov = cov * identity
    multi_mu = multiclass_mean(train_data, train_labels)
    densities = []
    for i in range(class_labels.size):
        densities.append(logLikelihood(test_data, vcol(multi_mu[i, :]), cov[i]))
    S = np.array(densities)
    logSJoint = S + np.log(prior_probability)
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = np.exp(logSPost)
    predictions = np.argmax(SPost, axis=0)

    if len(test_label) != 0:
        acc = 0
        for i in range(len(test_label)):
            if predictions[i] == test_label[i]:
                acc += 1
        acc /= len(test_label)

    return S, predictions, acc

def TiedGaussian(train_data, train_labels, test_data, prior_probability, test_label=[], final = 0):
    """
    Calculates the model of the Tied Gaussian classifier for a set of data, and applyes it to a test dataset
    :param train_date: matrix of the datapoints of a dataset used to train the model
    :param train_labels: row vector with the labels associated with each row of the training dataset
    :param test_data: matrix of the datapoints of a dataset used to test the model
    :param test_labels: row vector with the labels associated with each row of the test dataset
    :param prior_probability: col vector associated with the prior probability for each dimension
    :return S: matrix associated with the probability array
    :return predictions: Vector associated with the prediction of the class for each test data point
    :return acc: Accuracy of the validation set
    """
    class_labels = np.unique(train_labels)
    with_cov = covariance_within_class(train_data, train_labels)
    multi_mu = multiclass_mean(train_data, train_labels)
    densities = []
    for i in range(class_labels.size):
        densities.append(logLikelihood(test_data, vcol(multi_mu[i, :]), with_cov))
    S = np.array(densities)
    logSJoint = S + np.log(prior_probability)
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = np.exp(logSPost)
    predictions = np.argmax(SPost, axis=0)

    if len(test_label) != 0:
        acc = 0
        for i in range(len(test_label)):
            if predictions[i] == test_label[i]:
                acc += 1
        acc /= len(test_label)
    if final:
        return S, predictions, acc, multi_mu, with_cov
    else:
        return S, predictions, acc


def Tied_Naive_classifier(
    train_data, train_labels, test_data, prior_probability, test_label=[]
):
    """
    Calculates the model of the Tied Naive classifier for a set of data, and applyes it to a test dataset
    :param train_date: matrix of the datapoints of a dataset used to train the model
    :param train_labels: row vector with the labels associated with each row of the training dataset
    :param test_data: matrix of the datapoints of a dataset used to test the model
    :param test_labels: row vector with the labels associated with each row of the test dataset
    :param prior_probability: col vector associated with the prior probability for each dimension
    :return S: matrix associated with the probability array
    :return predictions: Vector associated with the prediction of the class for each test data point
    :return acc: Accuracy of the validation set
    """
    class_labels = np.unique(train_labels)
    cov = covariance_within_class(train_data, train_labels)
    identity = np.eye(cov.shape[1])
    cov = cov * identity
    multi_mu = multiclass_mean(train_data, train_labels)
    densities = []
    for i in range(class_labels.size):
        densities.append(logLikelihood(test_data, vcol(multi_mu[i, :]), cov))
    S = np.array(densities)
    logSJoint = S + np.log(prior_probability)
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = np.exp(logSPost)
    predictions = np.argmax(SPost, axis=0)

    if len(test_label) != 0:
        acc = 0
        for i in range(len(test_label)):
            if predictions[i] == test_label[i]:
                acc += 1
        acc /= len(test_label)
    return S, predictions, acc

def Generative_models(
    train_attributes,
    train_labels,
    test_attributes,
    prior_prob,
    test_labels,
    model,
    niter=0,
    alpha=0.1,
    threshold=10**-6,
    psi=0,
    final=0,
    quadratic=0,
):
    """

    Calculates the desired generative model
    :param train_date: matrix of the datapoints of a dataset used to train the model
    :param train_labels: row vector with the labels associated with each row of the training dataset
    :param test_data: matrix of the datapoints of a dataset used to test the model
    :param test_labels: row vector with the labels associated with each row of the test dataset
    :param prior_probability: col vector associated with the prior probability for each dimension
    :param: `model`defines which model, based on the following criterias:
        - `mvg`: Multivariate Gaussian Model
        - `Naive`: Naive Bayes Classifier
        - `Tied Gaussian`: Tied Multivariate Gaussian Model
        - `Tied naive`: Tied Naive Bayes Classifier
    :return S: matrix associated with the probability array
    :return predictions: Vector associated with the prediction of the class for each test data point
    :return acc: Accuracy of the validation set
    """
    if model.lower() == "mvg":
        if final:
            [Probabilities, Prediction, accuracy, mu, cov] = MVG_log_classifier(
                train_attributes,
                train_labels,
                test_attributes,
                prior_prob,
                test_labels,
                final,
            )
        else:
            [Probabilities, Prediction, accuracy] = MVG_log_classifier(
                train_attributes, train_labels, test_attributes, prior_prob, test_labels
            )
    elif model.lower() == "naive":
        [Probabilities, Prediction, accuracy] = Naive_log_classifier(
            train_attributes, train_labels, test_attributes, prior_prob, test_labels
        )
    elif model.lower() == "tied gaussian":
        if final:
            [Probabilities, Prediction, accuracy, mu, cov] = TiedGaussian(
            train_attributes, train_labels, test_attributes, prior_prob, test_labels, final=final
            )
        else:
            [Probabilities, Prediction, accuracy] = TiedGaussian(
            train_attributes, train_labels, test_attributes, prior_prob, test_labels
            )
            accuracy = round(accuracy * 100, 2)
    elif model.lower() == "tied naive":
        [Probabilities, Prediction, accuracy] = Tied_Naive_classifier(
            train_attributes, train_labels, test_attributes, prior_prob, test_labels
        )
        accuracy = round(accuracy * 100, 2)
    elif model.lower() == "gmm":
        if final:
            [Probabilities, Prediction, accuracy, mu, cov, w] = GMM(
                train_attributes,
                train_labels,
                test_attributes,
                test_labels,
                niter=niter,
                alpha=alpha,
                threshold=threshold,
                psi=psi,
                final=final,
            )
        else:
            [Probabilities, Prediction, accuracy] = GMM(
                train_attributes,
                train_labels,
                test_attributes,
                test_labels,
                niter=niter,
                alpha=alpha,
                threshold=threshold,
                psi=psi,
            )
    elif model.lower() == "diagonal":
        [Probabilities, Prediction, accuracy] = GMM(
            train_attributes,
            train_labels,
            test_attributes,
            test_labels,
            niter=niter,
            alpha=alpha,
            threshold=threshold,
            psi=psi,
            diag=1,
        )
    elif model.lower() == "tied":
        [Probabilities, Prediction, accuracy] = GMM(
            train_attributes,
            train_labels,
            test_attributes,
            test_labels,
            niter=niter,
            alpha=alpha,
            threshold=threshold,
            psi=psi,
            tied=1,
        )
    if final and (model.lower() == "mvg" or model.lower() == "tied gaussian"):
        return Probabilities, Prediction, accuracy, mu, cov
    elif final and model.lower() == "gmm":
        return Probabilities, Prediction, accuracy, mu, cov, w
    else:
        return Probabilities, Prediction, accuracy

def logreg_obj(v, DTR, LTR, l, pit):
    """
    Method to calculate the error of a function based on the data points
    :param v: Vector of the values to evaluate [w, b]
    :param DTR: Matrix with all the train attributes
    :param LTR: Matrix with all the train labels
    :param l: Hyperparameter l to apply to the function
    :return: retFunc the value of evaluating the function on the input parameter v
    """
   
    nt = np.count_nonzero(LTR == 1)
    nf = np.count_nonzero(LTR != 1)
    w, b = v[0:-1], v[-1]
    zi = LTR * 2 - 1
    
    non_target = DTR[:,LTR == 0]
    target = DTR[:,LTR == 1]

    log_sumt = 0
    log_sumf = 0
    zif = zi[zi == -1]
    zit = zi[zi == 1]
    
    inter_sol = -zit * (np.dot(w.T, target) + b)
    log_sumt = np.sum(np.logaddexp(0, inter_sol))

    inter_sol = -zif * (np.dot(w.T, non_target) + b)
    log_sumf = np.sum(np.logaddexp(0, inter_sol))
    retFunc = l / 2 * np.power(np.linalg.norm(w), 2) + pit / nt * log_sumt + (1-pit)/nf * log_sumf
   
    return retFunc

def binaryRegression(
    train_attributes,
    train_labels,
    l,
    test_attributes,
    test_labels,
    final=0,
    quadratic=0,
    pit=0.5
):
    """
    Method to calculate the error of a function based on the data points
    :param train_attributes: Matrix with all the train attributes
    :param train_labels: Matrix with all the train labels
    :param l: Hyperparameter l to apply to the function
    :param test_attributes: Matrix with all the train attributes
    :param test_labels: Matrix with all the train labels
    :return predictions: Vector associated with the prediction of the class for each test data point
    :return S: matrix associated with the probability array
    :return acc: Accuracy of the validation set
    """
    
    if quadratic == 1:
        xxt = np.dot(train_attributes.T, train_attributes).diagonal().reshape((1, -1))
        train_attributes = np.vstack((xxt, train_attributes))

        zzt = np.dot(test_attributes.T, test_attributes).diagonal().reshape((1, -1))
        test_attributes = np.vstack((zzt, test_attributes))
    x0 = np.zeros(train_attributes.shape[0] + 1)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(
        logreg_obj, x0, approx_grad=True, args=(train_attributes, train_labels, l, pit)
    )
    w, b = x[0:-1], x[-1]
    S = np.dot(w.T, test_attributes) + b
    funct = lambda s: 1 if s > 0 else 0
    predictions = np.array(list(map(funct, S)))
    acc = 0
    for i in range(test_labels.shape[0]):
        if predictions[i] == test_labels[i]:
            acc += 1
    acc /= test_labels.size
    acc = round(acc * 100, 2)
    if final:
        return predictions, S, acc, w, b
    else:
        return predictions, S, acc

def k_fold(
    k,
    attributes,
    labels,
    previous_prob,
    model="mvg",
    PCA_m=0,
    LDA_m=0,
    l=0.001,
    pi=[0.5],
    Cfn=1,
    Cfp=1,
    final=0,
    quadratic=0,
    pit=0.5,
    niter=0,
    alpha=0,
    psi=0,
    C = 1,
    gamma=1,
    dim = 2,
    c=1,
    seed = 0,
    score_calibration_flag = 0,
):
    """
    Applies a k-fold cross validation on the dataset, applying the specified model.
    :param: `k` Number of partitions to divide the dataset
    :param: `attributes` matrix containing the whole training attributes of the dataset
    :param: `labels` the label vector related to the attribute matrix
    :param: `previous_prob` the column vector related to the prior probability of the dataset
    :param: `model` (optional). Defines the model to be applied to the model:
        - `mvg`: Multivariate Gaussian Model
        - `Naive`: Naive Bayes Classifier
        - `Tied Gaussian`: Tied Multivariate Gaussian Model
        - `Tied naive`: Tied Naive Bayes Classifier
        - `Regression` : Binomial Regression
        - `SVM`: Support Vector Machine (default)
        - `polynomial` : Support Vector Machine with Polynomial Kernel model
        - `radial` : Support vector Machine with Radial Kernel model
    :param: `PCA_m` (optional) a number of dimensions to reduce using PCA method
    :param: `LDA_m` (optional) a number of dimensions to reduce using LDA mehtod
    :param: `l` (optional) hyperparameter to use when the method is linera regression, default value set to 0.001
    :param: `pi` (optional) hyperparameter to use when calculating the Bayes risk, default value set to 0.5
    :return final_acc: Accuracy of the validation set
    :return final_S: matrix associated with the probability array
    """
    section_size = int(attributes.shape[1] / k) if attributes.ndim > 1 else int(attributes.size / k)
    low = 0
    all_values = np.c_[attributes.T, labels]
    all_values = np.random.default_rng(seed=seed).permutation(all_values)
    attributes = all_values[:, 0:all_values.shape[1] - 1].T
    labels = all_values[:, -1].astype("int32")
    high = section_size
    model = model.lower()
    minDCF_dic = {}
    final_S = {}
    for p in pi:
        minDCF_dic[p] = 0
    for i in range(k):
        if not i:
            validation_att = attributes[:, low:high]
            validation_labels = labels[low:high]
            train_att = attributes[:, high:]
            train_labels = labels[high:]
            if PCA_m:
                P, train_att = PCA(train_att, PCA_m)
                validation_att = np.dot(P.T, validation_att)
                if LDA_m:
                    W, _ = LDA1(train_att, train_labels, LDA_m)
                    train_att = np.dot(W.T, train_att)
                    validation_att = np.dot(W.T, validation_att)
            if model == "regression":
                if final:
                    [prediction, S, acc, w, b] = binaryRegression(
                        train_att,
                        train_labels,
                        l,
                        validation_att,
                        validation_labels,
                        final=1,
                        quadratic=quadratic,
                        pit=pit
                    )
                    final_w = w
                    final_b = b
                else:
                    [prediction, S, acc] = binaryRegression(
                        train_att,
                        train_labels,
                        l,
                        validation_att,
                        validation_labels,
                        quadratic=quadratic,
                        pit=pit
                    )
            elif model == "svm" or model == "polynomial" or model == "radial":
                if final:
                    final_x = np.zeros((attributes.shape[1]))
                    [S, prediction, acc, x] = svm(
                        train_att, train_labels, validation_att, validation_labels, C, model=model, gamma=gamma, dim=dim, c=c, final=1)
                    final_x[high:] += x
                else:
                    [S, prediction, acc] = svm(train_att, train_labels, validation_att, validation_labels, C, model=model, gamma=gamma, dim=dim, c=c)

            else:
                if final and model != "gmm":
                    [S, prediction, acc, mu, cov] = Generative_models(
                        train_att,
                        train_labels,
                        validation_att,
                        previous_prob,
                        validation_labels,
                        model,
                        niter=niter,
                        psi=psi,
                        alpha=alpha,
                        final=final,
                    )
                    final_w = P
                    final_mu = mu
                    final_cov = cov
                elif final:
                    [S, prediction, acc, mu, cov, w] = Generative_models(
                    train_att,
                    train_labels,
                    validation_att,
                    previous_prob,
                    validation_labels,
                    model,
                    niter=niter,
                    psi=psi,
                    alpha=alpha,
                    final=final,
                )
                    final_mu = mu
                    final_cov = cov
                    final_w = w
                else:
                    [S, prediction, acc] = Generative_models(
                        train_att,
                        train_labels,
                        validation_att,
                        previous_prob,
                        validation_labels,
                        model,
                        niter=niter,
                        psi=psi,
                        alpha=alpha
                    )
            for p in pi:
                (minDCF, _, _, _) = minCostBayes(S, validation_labels, p, Cfn, Cfp)
                minDCF_dic[p] += minDCF
            final_S = S
            final_acc = acc
            final_predictions = prediction
            continue
        low += section_size
        high += section_size
        ## Check if the last section is not complete or if it is the last section
        if high > attributes.shape[1] or attributes.shape[1] - high < section_size:
            high = attributes.shape[1]
        validation_att = attributes[:, low:high]
        validation_labels = labels[low:high]
        train_att = attributes[:, :low]
        train_labels = labels[:low]
        train_att = np.hstack((train_att, attributes[:, high:]))
        train_labels = np.hstack((train_labels, labels[high:]))
        if PCA_m:
            P, train_att = PCA(train_att, PCA_m)
            validation_att = np.dot(P.T, validation_att)
            if LDA_m:
                W, _ = LDA1(train_att, train_labels, LDA_m)
                train_att = np.dot(W.T, train_att)
                validation_att = np.dot(W.T, validation_att)
        if model == "regression":
            if final:
                [prediction, S, acc, w, b] = binaryRegression(
                    train_att,
                    train_labels,
                    l,
                    validation_att,
                    validation_labels,
                    final=1,
                    quadratic=quadratic,
                    pit=pit
                )
                final_w += w
                final_b += b
                if LDA_m:
                    final_LDA += W
            else:
                [prediction, S, acc] = binaryRegression(
                    train_att,
                    train_labels,
                    l,
                    validation_att,
                    validation_labels,
                    quadratic=quadratic,
                    pit=pit
                )
        elif model == "svm" or model == "polynomial" or model == "radial":
            if final:
                [S, prediction, acc, x] = svm(train_att, train_labels, validation_att, validation_labels, C, model=model, gamma=gamma, dim=dim, c=c, final=1)
                final_x[:low] += x[:low]
                final_x[high:] += x[low:]   
            else:
                [S, prediction, acc] = svm(
                    train_att, train_labels, validation_att, validation_labels, C, model=model, gamma=gamma, dim=dim, c=c) 
        else:
            if final and model.lower() != "gmm":
                [S, prediction, acc, mu, cov] = Generative_models(
                    train_att,
                    train_labels,
                    validation_att,
                    previous_prob,
                    validation_labels,
                    model,
                    niter=niter,
                    psi=psi,
                    alpha=alpha,
                    final=final,
                )
                final_mu += mu
                final_cov += cov
                final_w += P
            elif final:
                [S, prediction, acc, mu, cov, w] = Generative_models(
                    train_att,
                    train_labels,
                    validation_att,
                    previous_prob,
                    validation_labels,
                    model,
                    niter=niter,
                    psi=psi,
                    alpha=alpha,
                    final=final,
                )
                final_mu += mu
                final_cov += cov
                final_w += w
            else:
                [S, prediction, acc] = Generative_models(
                    train_att,
                    train_labels,
                    validation_att,
                    previous_prob,
                    validation_labels,
                    model,
                    niter=niter,
                    psi=psi,
                    alpha=alpha
                )
        for p in pi:
            (minDCF, _, _, _) = minCostBayes(S, validation_labels, p, Cfn, Cfp)
            minDCF_dic[p] += minDCF
        final_S = np.hstack((final_S, S))
        final_acc += acc
        final_predictions = np.append(final_predictions, prediction)
    final_acc = round(final_acc / k, 4)
    for p in pi:
        minDCF_dic[p] = round(minDCF_dic[p] / k, 4)
    if model == "regression" and final:
        final_w /= k
        final_b /= k
        return (
            final_S,
            prediction,
            minDCF_dic,
            final_w,
            final_b
        )
    elif (model == "svm" or model == "polynomial" or model == "radial") and final:
        final_x /= k
        return final_S, minDCF_dic, attributes, labels, final_x
    elif final:
        final_mu /= k
        final_cov /= k
        final_w /= k
        return (
            final_S,
            labels,
            minDCF_dic,
            final_mu,
            final_cov,
            final_w,
        )
    elif score_calibration_flag:
        return final_S, labels, minDCF_dic
    else:
        return final_S, prediction, final_acc, minDCF_dic

def EM(x, mu, cov, w, threshold=10**-6, psi=0, diag=0, tied=0):
    delta = 100
    previous_ll = 10
    cont = 1000
    mu = np.array(mu)
    cov = np.array(cov)
    if diag:
        cov = cov * np.eye(cov.shape[1])
    if tied:
        cov[:] = np.sum(cov, axis=0) / x.shape[1]
    if psi:
        for i in range(cov.shape[0]):
            U, s, _ = np.linalg.svd(cov[i])
            s[s < psi] = psi
            cov[i] = np.dot(U, vcol(s) * U.T)
    w = vcol(np.array((w)))
    while (delta > threshold) and (cont > 0):
        #### E-STEP ####
        ll = np.array(ll_gaussian(x, mu, cov))
        SJoint = ll + np.log(w)
        logSMarginal = scipy.special.logsumexp(SJoint, axis=0)
        logSPost = SJoint - logSMarginal
        SPost = np.exp(logSPost)

        #### M - STEP ####
        fg = np.dot(SPost, x.T)
        zg = vcol(np.sum(SPost, axis=1))
        sg = []
        n_mu = fg / zg
        new_C = []
        mul = []
        for i in range(mu.shape[0]):
            psg = np.zeros((x.shape[0], x.shape[0]))
            for j in range(x.shape[1]):
                xi = x[:, j].reshape((-1, 1))
                xii = np.dot(xi, xi.T)
                psg += SPost[i, j] * xii
            mul.append(np.dot(vcol(n_mu[i, :]), vcol(n_mu[i, :]).T))
            sg.append(psg)
        div = np.array(sg) / zg.reshape((-1, 1, 1))
        new_mu = np.array(n_mu)
        mul = np.array(mul)
        new_C = div - mul
        new_w = vcol(zg / np.sum(zg, axis=0))
        if diag:
            new_C = new_C * np.eye(new_C.shape[1])
        if tied:
            new_C[:] = np.sum(new_C, axis=0) / x.shape[1]
        if psi:
            for i in range(new_C.shape[0]):
                U, s, _ = np.linalg.svd(new_C[i])
                s[s < psi] = psi
                new_C[i] = np.dot(U, vcol(s) * U.T)
        previous_ll = np.sum(logSMarginal) / x.shape[1]
        s = np.array(ll_gaussian(x, new_mu, new_C))
        newJoint = s + np.log(new_w)
        new_marginal = scipy.special.logsumexp(newJoint, axis=0)
        avg_ll = np.sum(new_marginal) / x.shape[1]
        delta = abs(previous_ll - avg_ll)
        previous_ll = avg_ll
        mu = new_mu
        cov = new_C
        w = new_w
        cont -= 1
        # print(delta)
    return avg_ll, mu, cov, w

def LBG(x, niter, alpha, psi=0, diag=0, tied=0):
    mu = mean_of_matrix_rows(x)
    mu = mu.reshape((1, mu.shape[0], mu.shape[1]))
    C = covariance(x)
    C = C.reshape((1, C.shape[0], C.shape[1]))
    w = np.ones(1).reshape(-1, 1, 1)
    if not niter:
        [ll, mu, C, w] = EM(x, mu, C, w, psi=psi, diag=diag, tied=tied)
        mu = mu.reshape((-1, mu.shape[1], 1))
        w = w.reshape((-1, 1, 1))
    new_gmm = []
    for i in range(niter):
        new_mu = []
        new_cov = []
        new_w = []
        for i in range(len(mu)):
            U, s, _ = np.linalg.svd(C[i])
            d = U[:, 0:1] * s[0] ** 0.5 * alpha
            new_w.append(w[i] / 2)
            new_w.append(w[i] / 2)
            new_mu.append(mu[i] + d)
            new_mu.append(mu[i] - d)
            new_cov.append(C[i])
            new_cov.append(C[i])
        [ll, mu, C, w] = EM(x, new_mu, new_cov, new_w, psi=psi, diag=diag, tied=tied)
        mu = mu.reshape((-1, mu.shape[1], 1))
        w = w.reshape((-1, 1, 1))
    # print(gmm)
    return ll, mu, C, w

def GMM(
    train_data,
    train_labels,
    test_data,
    test_label,
    niter,
    alpha,
    threshold,
    psi=0,
    diag=0,
    tied=0,
    final=0,
):
    class_labels = np.unique(train_labels)
    cov = multiclass_covariance(train_data, train_labels)
    densities = []
    class_mu = []
    class_c = []
    class_w = []
    for i in class_labels:
        [_, mu, cov, w] = LBG(
            train_data[:, train_labels == i],
            niter=niter,
            alpha=alpha,
            psi=psi,
            diag=diag,
            tied=tied,
        )
        class_mu.append(mu)
        class_c.append(cov)
        class_w.append(w)
    class_mu = np.array(class_mu)
    class_c = np.array(class_c)
    class_w = np.array(class_w)
    densities = []
    for i in class_labels:
        ll = np.array(ll_gaussian(test_data, class_mu[i], class_c[i]))
        Sjoin = ll + np.log(class_w[i].reshape((-1, 1)))
        logdens = scipy.special.logsumexp(Sjoin, axis=0)
        densities.append(logdens)
    S = np.array(densities)
    predictions = np.argmax(S, axis=0)
    if len(test_label) != 0:
        acc = 0
        for i in range(len(test_label)):
            if predictions[i] == test_label[i]:
                acc += 1
        acc /= len(test_label)
        acc = round(acc * 100, 2)
        
    if final:
        return S, predictions, acc, class_mu, class_c, class_w
    else:
        return S, predictions, acc

def minCostBayes(llr, labels, pi, Cfn, Cfp):
    if llr.ndim > 1:
        llr = llr[1, :] - llr[0,:]
    sortedLLR = np.sort(llr)
    t = np.array([-np.inf, np.inf])
    t = np.append(t, sortedLLR)
    t = np.sort(t)
    DCFnorm = []
    FNRlist = []
    FPRlist = []
    threashols = []
    for i in t:
        funct = lambda s: 1 if s > i else 0
        decisions = np.array(list(map(funct, llr)))

        tp = np.sum(np.logical_and(decisions == 1, labels == 1))
        fp = np.sum(np.logical_and(decisions == 1, labels == 0))
        tn = np.sum(np.logical_and(decisions == 0, labels == 0))
        fn = np.sum(np.logical_and(decisions == 0, labels == 1))

        confusion_matrix = np.array([[tn, fn], [fp, tp]])
        M01 = confusion_matrix[0][1]
        M11 = confusion_matrix[1][1]
        M10 = confusion_matrix[1][0]
        M00 = confusion_matrix[0][0]

        if M00 == 0 and M10 == 0:
            FNR = 1
            FPR = 0
        elif M11 == 0 and M01 == 0:
            FNR = 0
            FPR = 1
        else:
            FPR = M10 / (M00 + M10)
            FNR = M01 / (M01 + M11)

        DCF = pi * Cfn * FNR + (1 - pi) * Cfp * FPR
        B = min(pi * Cfn, (1 - pi) * Cfp)
        DCFnormal = DCF / B

        FNRlist = np.append(FNRlist, FNR)
        FPRlist = np.append(FPRlist, FPR)
        DCFnorm = np.append(DCFnorm, DCFnormal)
        threashols = np.append(threashols, i)
    minDCF = min(DCFnorm)
    minT = threashols[np.where( DCFnorm == minDCF)]

    return minDCF, FPRlist, FNRlist, minT

def polynomial_kernel(xi, xj, d, C, eps):
    interm = np.dot(xi.T, xj)
    interm += C
    G = np.power(interm, d) + eps
    return G


def radial_kernel(xi, xj, gamma, eps):
    diff = xi[:, :, np.newaxis] - xj[:, np.newaxis, :]
    G = -gamma * np.square(np.linalg.norm(diff, axis=0))
    G = np.add(np.exp(G), eps)
    return G


def dual_svm(
    alpha,
    training_att,
    training_labels,
    K=1,
    d=0,
    c=0,
    eps=0,
    gamma=0,
    funct="polynomial",
):
    kern = funct.lower()
    one = np.ones(training_att.shape[1])
    zi = 2 * training_labels - 1
    if kern == "polynomial":
        G = polynomial_kernel(training_att, training_att, d, c, eps)
    elif kern == "radial":
        G = radial_kernel(training_att, training_att, gamma, eps=eps)
    else:
        D = np.vstack((training_att, one * K))
        G = np.dot(D.T, D)
    z = np.outer(zi, zi)
    H = np.multiply(z, G)
    retFun = np.dot(alpha.T, H)
    retFun = np.dot(retFun, alpha) / 2
    retFun = retFun - np.dot(alpha.T, one)
    retGrad = np.dot(H, alpha)
    retGrad -= one
    return (retFun, retGrad)


def svm(
    training_att,
    training_labels,
    test_att,
    test_labels,
    constrain,
    model="",
    dim=2,
    c=1,
    K=1,
    gamma=1,
    eps=0,
    final=0,
    pit=0.5,
    pitemp=0.7
):
    """
    Apply the Support Vector Machine model, using either one of the models described to approximate the soluction.
    :param train_att: Matrix with all the train attributes
    :param train_labels: Matrix with all the train labels
    :param test_att: Matrix with all the train attributes
    :param test_labels: Matrix with all the train labels
    :param constrain: Constrain of maximum value of the alpha vector
    :param model: (optional) define the applied kernel model:
    - `polynomial`: polynomial kernel of degree d
    - `Radial`: Radial Basis Function kernel
    - Default: leave empty, and the dual SVM method is applied
    :param dim: (optional) hyperparameter for polynomial method. `Default 2`
    :param c: (optional) hyperparameter for polynomial method. `Default 1`
    :param K: (optional) hyperparameter for dual SVM method. `Default 1`
    :param gamma: (optional) hyperparameter for radial method. `Default 1`
    :param eps: (optional) hyperparameter for kernel methods. `Default 0`
    """
    alp = np.ones(training_att.shape[1])
    constrain = np.array([(0, constrain)] * training_att.shape[1])
    TL=np.array(training_labels)
    TL = TL.reshape(-1, 1)
    pitempM=1-pitemp
    pitempF=pitemp
    pitM=1-pit
    pitF=pit
    constrain= np.where(TL==0, constrain*(pitM/pitempM), constrain*(pitF/pitempF))
    [x, _, _] = scipy.optimize.fmin_l_bfgs_b(
        dual_svm,
        alp,
        args=(training_att, training_labels, K, dim, c, eps, gamma, model),
        bounds=constrain,
        factr=1000000000   
    )
    zi = 2 * training_labels - 1
    kern = model.lower()
    if kern == "polynomial":
        S = x * zi
        S = np.dot(S, polynomial_kernel(training_att, test_att, dim, c, eps))
    elif kern == "radial":
        S = x * zi
        S = np.dot(S, radial_kernel(training_att, test_att, gamma, eps))
    else:
        D = np.ones(training_att.shape[1]) * K
        D = np.vstack((training_att, D))
        w = x * zi
        w = w * D
        w = np.sum(w, axis=1)
        x_val = np.vstack((test_att, np.ones(test_att.shape[1]) * K))
        S = np.dot(w.T, x_val)
    predictions = np.where(S > 0, 1, 0)
    error = np.abs(predictions - test_labels)
    error = np.sum(error)
    error /= test_labels.size
    if final:
        return S, predictions, 1 - error, x
    else:
        return S, predictions, 1 - error

def histogram_1n(non_target, target, x_axis="", y_axis="", legend = 1, alpha_val = 0.5):
    plt.xlim((min(min(non_target), min(target)) - 1, max(max(non_target), max(target)) + 1))
    plt.hist(non_target, color="blue", alpha=alpha_val, label=class_label[0], density=True, bins=50, edgecolor='grey')
    plt.hist(
        target, color="red", alpha=alpha_val, label=class_label[1], density=True, bins=20, edgecolor='grey'
    )
    if legend:
        plt.legend(class_label)

def scatter_2d(non_target, target, x_axis="", y_axis=""):
    plt.scatter(non_target[0], non_target[1], c="blue", s=1.5)
    plt.scatter(target[0], target[1], c="red", s=1.5)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)


def graficar(attributes, labels, save = 0):
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
                if save:
                    plt.savefig(f"{os.getcwd()}/Image/histogram-dim-{cont}.png")
                plt.show()
                cont += 1

def long_graficar(attributes, labels):
    attribute_names = []
    for i in range(attributes.shape[0]):
        attribute_names.append(str(i))
    values_histogram = {}

    for i in range(len(attribute_names)):
        values_histogram[attribute_names[i]] = [
            attributes[i, labels == 0],
            attributes[i, labels == 1],
        ]

    for a in attribute_names:
        histogram_1n(
            values_histogram[a][0],
            values_histogram[a][1],
            x_axis=a,
        )


    cont = 1
    for xk, xv in values_histogram.items():
        for yk, yv in values_histogram.items():
            if xk == yk:
                plt.subplot(attributes.shape[0], attributes.shape[0], cont)
                histogram_1n(xv[0], xv[1], legend=0)
                cont += 1
            else:
                plt.subplot(attributes.shape[0], attributes.shape[0], cont)
                scatter_2d([xv[0], yv[0]], [xv[1], yv[1]])
                cont += 1
    plt.show()

def graf_LDA(attributes, lables, save=False):

    W, _ = LDA1(attributes, lables, 1)
    LDA_attributes = np.dot(W.T, attributes)
    print(LDA_attributes.shape)
    histogram_1n(LDA_attributes[0, lables==0], LDA_attributes[0, lables==1])
    plt.title("LDA Direction")
    if save:
        plt.savefig(f"{os.getcwd()}/Image/LDA-direction.png")
    plt.show()

def graf_PCA(attributes, lables):
    fractions = []
    total_eig, _ = np.linalg.eigh(covariance(attributes))
    total_eig = np.sum(total_eig)
    for i in range(attributes.shape[0]):
        if i == 0:
            continue
        _, reduces_attributes = PCA(attributes, i)
        PCA_means, _  = np.linalg.eigh(covariance(reduces_attributes))
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
    #correlation of non-target language
    nont_language = attributes[:, labels==0]
    corr_attr = np.corrcoef(nont_language)
    sns.heatmap(corr_attr, cmap='Blues')
    plt.title(f"{class_label[0]} attribute correlation")
    plt.savefig(f"{os.getcwd()}/Image/{class_label[0]}-correlation.png")
    plt.show()
    #correlation of target language
    targ_language= attributes[:, labels==1]
    corr_attr = np.corrcoef(targ_language)
    sns.heatmap(corr_attr, cmap="Reds")
    plt.title(f"{class_label[1]} attribute correlation")
    plt.savefig(f"{os.getcwd()}/Image/{class_label[1]}-correlation.png")
    plt.show()

def gaussian_train(attributes, labels, headers, priorProb = 0, pi=[0.5], pca=[5]):
    """
    Trains a list of Gaussian models for a dataset, and prints the results of the training
    """
    ### Parameter definition ###
    tableKFold = []
    priorProb = vcol(np.ones(2) * 0.5) if not priorProb else priorProb
    k_fold_value = 6
    ####
    tableKFold.append(["Full"])
    c2 = 1
    list_minDCF = []
    for model in headers:
        tableKFold[0].append([])

        [_, _, _, minDCF] = k_fold(
                k_fold_value, attributes, labels, priorProb, model=model, pi=pi
            )
        for p in pi:
            if p == 0.5:
                minDCF[p] = (minDCF[0.1] + minDCF[p]) / 2
            tableKFold[0][c2].append([minDCF[p]])
        list_minDCF.append(minDCF)
        c2 += 1

    cont = 1
    for i in pca:
        tableKFold.append([f"PCA {i}"])
        c2 = 1
        for model in headers:
            tableKFold[cont].append([])
            [_, _, _, minDCF] = k_fold(
                    k_fold_value,
                    attributes,
                    labels,
                    priorProb,
                    pi=pi,
                    model=model,
                    PCA_m=i,
                )
            for p in pi:
                if p == 0.5:
                    minDCF[p] = (minDCF[0.1] + minDCF[p]) / 2
                tableKFold[cont][c2].append([minDCF[p]])
                list_minDCF.append(minDCF)
            c2 += 1
        cont += 1

    newHeaders = []
    for i in headers:
        newHeaders.append(i + " MinDCF" + " ".join(str(p) for p in pi))
    print(tabulate(tableKFold, headers=newHeaders))

def regression_train(attributes, labels, lamda, pi=[0.5], quadratic = 0, k = 5, pca = 0, priorProb = [0.5, 0.5]):
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
        [_, _, accuracy, minDCF] = k_fold(
                k,
                attributes,
                labels,
                priorProb,
                "regression",
                l=l,
                pi=pi,
                quadratic=quadratic,
                PCA_m=pca
            )
        for p in pi:
            if p== 0.1:
                minDCFvalues1.append(minDCF[p])
            if p== 0.2:
                minDCFvalues2.append(minDCF[p])
            if p== 0.5:
                val = (minDCF[p] + minDCFvalues1[-1])/2
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

def svm_train(attributes, labels, C, model= "svm", pi=[0.5], k = 5, pca = 0, z =0):
    minDCFvalues1=[]
    minDCFvalues2=[]
    minDCFvalues5=[]
    title = "Support Vector Machine"
    title += f"with kernel model {model}." if model != "svm" else "."
    title = title + f"PCA: {pca}" if pca else title
    for c in C:
        [_, _, _, minDCF] = k_fold(
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
        [_, _, _, minDCF] = k_fold(
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
                [_, _, _, minDCF] = k_fold(
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
            [S, _, _, minDCF] = k_fold(
            k_value, attributes, labels, prior_probability, model=mod, niter=constrains[1],alpha=constrains[0], psi=constrains[2], pi=pi
            )
        else:
            [S, _, _, minDCF] = k_fold(
                k_value, attributes, labels, prior_probability, model=mod, niter=constrains[1],alpha=constrains[0], psi=constrains[2], PCA_m=m, pi=pi
                )
            
        for p in pi:
            if p == 0.5:
                minDCF[p] = (minDCF[0.1] + minDCF[p]) / 2
                result_minDCF.append(minDCF[p])
            tableGMM[cont][c2].append([minDCF[p]])
            perc += 1
            list_minDCF.append(minDCF)
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
    plt.show()

def OptimalBayes(llr, labels, pi, Cfn, Cfp):
    if llr.ndim > 1:
        llr = llr[1, :] - llr[0,:]
    threshold = -np.log((pi * Cfn) / ((1 - pi) * Cfp))
    funct = lambda s: 1 if s > threshold else 0
    decisions = np.array(list(map(funct, llr)))

    tp = np.sum(np.logical_and(decisions == 1, labels == 1))
    fp = np.sum(np.logical_and(decisions == 1, labels == 0))
    tn = np.sum(np.logical_and(decisions == 0, labels == 0))
    fn = np.sum(np.logical_and(decisions == 0, labels == 1))

    confusion_matrix = np.array([[tn, fn], [fp, tp]])

    return confusion_matrix, decisions

def Bayes_risk(llr, labels, pi, Cfn, Cfp):
    [confusion_matrix, _] = OptimalBayes(llr, labels, pi, Cfn, Cfp)
    M01 = confusion_matrix[0][1]
    M11 = confusion_matrix[1][1]
    M10 = confusion_matrix[1][0]
    M00 = confusion_matrix[0][0]
    if M00 == 0 and M10 == 0:
        FNR = 1
        FPR = 0
    elif M11 == 0 and M01 == 0:
        FNR = 0
        FPR = 1
    else:
        FPR = M10 / (M00 + M10)
        FNR = M01 / (M01 + M11)

    DCF = pi * Cfn * FNR + (1 - pi) * Cfp * FPR

    B = min(pi * Cfn, (1 - pi) * Cfp)

    DCFnorm = DCF / B

    return DCF, min(DCFnorm, 1)

def BayesErrorPlot(llr, labels, Cfn, Cfp, model=""):
    DCFlist = []
    minDCFlist = []
    effPriorLogOdds = np.linspace(-5, 5, 15)
    prior = 1 / (1 + np.exp(-effPriorLogOdds))

    for i in prior:
        (_, DCFnorm) = Bayes_risk(llr, labels, i, Cfn, Cfp)
        (minDCF, _, _, _) = minCostBayes(llr, labels, i, Cfn, Cfp)
        DCFlist = np.append(DCFlist, DCFnorm)
        minDCFlist = np.append(minDCFlist, minDCF)

    plt.figure()
    plt.plot(effPriorLogOdds, DCFlist, label="DCF", color="r")
    plt.plot(effPriorLogOdds, minDCFlist, label="min DCF", color="b", linestyle="--")
    plt.ylim([0, 1.5])
    plt.xlim([-5, 5])
    plt.ylabel("DCF")
    plt.title(f"Bayes Error Plot {model}")
    plt.grid()
    plt.legend()
    plt.show()

def ROCcurve(FPRlist, FNRlist, model = ""):
    TPR = 1 - FNRlist
    plt.plot(FPRlist, TPR, label=model)

def det_plot(FPRlist, FNRlist, color='b', model = "", show=True):
    """plots the ROC curve for the given scores and labels, optionally does not create another figure to enable multiple plots"""

      
    plt.plot(FPRlist, FNRlist, color=color, label=model)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("FPR")
    plt.ylabel("FNR")
    plt.grid(True)
    if(show):
        plt.show()

def calculate_logreg(test_points, w, b, quadratic = 0):
    if quadratic:
        xt_2 = np.dot(test_points.T, test_points).diagonal().reshape((1, -1))
        test_points = np.vstack((xt_2, test_points))
    S = np.dot(w.T, test_points) + b
    return S

def calculate_svm(x, zi, train_att, test_feat, args):
    S = x * zi
    if args["model"]=="polynomial":
        S = np.dot(S, polynomial_kernel(train_att, test_feat, args["dim"], args["c"], args["epsilon"]))
    elif args["model"]=="radial":
        S = np.dot(S, radial_kernel(train_att, test_feat, args["gamma"], args["epsilon"]))
    return S

def calculate_gmm(test_attributes, test_labels, class_mu, class_c, class_w):
    densities = []
    for i in np.unique(test_labels):
        ll = np.array(ll_gaussian(test_attributes, class_mu[i], class_c[i]))
        Sjoin = ll + np.log(class_w[i].reshape((-1, 1)))
        logdens = scipy.special.logsumexp(Sjoin, axis=0)
        densities.append(logdens)
    return np.array(densities)

def linear_regression_2C(feat, labels, l):
    x0 = np.zeros(feat.shape[0] + 1)
    
    opt,_1,_2 = fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, args=(feat, labels, l, labels.shape[0]))
    wopt, bopt = opt[0:-1], opt[-1]
    
    return wopt, bopt

def score_calibration(K, DTR, LTR, l, LOO=False, testing = False):  
    """
    Calibrates the incoming data using the linear regression model with a k-fold cross validation
    """
    if DTR.ndim == 1:
        DTR = vrow(DTR)
    N = DTR.shape[1]
    if(LOO):
        K = N
    seglen = N//K
    alls = []
    if testing:
        wopt, bopt = 0, 0
    for i in range(K):
        mask = [True] * N
        mask[i*seglen : (i+1)*seglen] = [False] * seglen
        notmask = [not x for x in mask]
        DTRr = DTR[:,mask]
        LTRr = LTR[mask]
        DTV = DTR[:,notmask]  
        wopt, bopt = linear_regression_2C(DTRr, LTRr, l)
        scores = np.dot(vrow(wopt), DTV) + bopt
        if testing:
            wopt += wopt
            bopt += bopt
        alls += scores.ravel().tolist()
    return alls, wopt, bopt if testing else alls


if __name__ == "__main__":
    
    ## Flags for running the code
    data_vizualization = 0 # Run the code corresponding to the data visualization
    gaussian_model = 0 # Run the code corresponding to the Gaussian model training
    log_reg = 0 # Run the code corresponding to the logistic regression model training
    svm_flag = 0 # Run the code corresponding to the SVM model training
    gmm_flag = 0 # Run the code corresponding to the GMM model training
    score_cal = 0 # Run the code corresponding to the score calibration
    evaluation_flag = 1 # Run the code corresponding to the evaluation of the models

    ## Load the data
    path = os.path.abspath("data/Test.txt")
    [full_train_att, full_train_labels] = load(path)
    standard_deviation = np.std(full_train_att)
    z_data = center_data(full_train_att) / standard_deviation

    pathTest = os.path.abspath("Data/Test.txt")
    [full_test_att, full_test_labels] = load(pathTest)
    standar_test = np.std(full_test_att)
    z_test = center_data(full_test_att) / standar_test

    ## Hyperparameters for all the code
    pi = [0.1, 0.2, 0.5]
    k = 5 # Number of iteration for the k cross validation
    priorProb = vcol(np.ones(2) * 0.5) # Prior probability for the Gaussian model
    Cfn = 1 # Cost of false negative
    Cfp = 1 # Cost of false positive

    if data_vizualization:
        ## Print the data
        for i in range(full_train_att.shape[0]):
            print(f"max {i}: {max(full_train_att[i])}", end="\t")
            print(f"min {i}: {min(full_train_att[i])}")
        print(f"Attribute dimensions: {full_train_att.shape[0]}")
        print(f"Points on the dataset: {full_train_att.shape[1]}")
        print(
            f"Distribution of full_train_labels (0, 1): {full_train_labels[full_train_labels==0].shape}, {full_train_labels[full_train_labels==1].shape}"
        )
        ## Graph each dimension of the data
        graficar(full_train_att, full_train_labels, save=0)
        ## Graph the LDA and PCA analysis
        graf_LDA(full_train_att, full_train_labels)
        graf_PCA(full_train_att, full_train_labels)
        ## Graph the correlation of the data
        graph_corr(full_train_att, full_train_labels)
        ## Graph the whole attributes compared one to another
        long_graficar(full_train_att, full_train_labels)
    
    if gaussian_model:
        models = ["MVG", "Naive", "Tied Gaussian", "Tied Naive"]
        print("Training Gaussian models for the full dataset")
        gaussian_train(full_train_att, full_train_labels, models, pi=pi)
        print("Training Gaussian models for the Z-norm dataset")
        gaussian_train(z_data, full_train_labels, models,pi=pi)
    
    if log_reg:
        l_list = np.logspace(-6, 6, 10, base=10)
        ## LOGISTIC REGRESSIONS ##
        print("Training Logistic Regression")
        regression_train(full_train_att, full_train_labels, l_list, pi=pi, quadratic=0, priorProb=priorProb) # Full dataset
        regression_train(z_data, full_train_labels, l_list, pi=pi, quadratic=0, priorProb=priorProb) # Z-norm dataset
        regression_train(full_train_att, full_train_labels, l_list, pi=pi, pca=5, quadratic=0, priorProb=priorProb) # With PCA 5 applied


        ## QUADRATIC REGRESSIONS ##   
        print("Training Quadratic Regression") 
        regression_train(full_train_att, full_train_labels, l_list,pi=pi, quadratic=1, priorProb=priorProb)
        regression_train(z_data, full_train_labels,l_list, pi=pi, quadratic=1, priorProb=priorProb) # Z-norm dataset
        regression_train(full_train_att, full_train_labels, l_list, pi=pi, pca=5, quadratic=1, priorProb=priorProb) # With PCA 5 applied
    
    if svm_flag:
        initial_C = np.logspace(-6, 6, 10)
        gamma = np.logspace(-3,3,10)
        k = 5
        print("Training SVM models")
        # ### -------full dataset ----- ###
        print("Training full SVM")
        svm_train(full_train_att, full_train_labels, initial_C, pi=pi)
        svm_train(full_train_att, full_train_labels, initial_C, pi=pi, pca=5)
        print("Training Polynomial")
        svm_train(full_train_att, full_train_labels, initial_C, pi=pi, model="polynomial")
        svm_train(full_train_att, full_train_labels, initial_C, pi=pi, model="polynomial", pca=5)
        print("Trianing Radial")
        svm_train(full_train_att, full_train_labels, initial_C, pi=pi, model="radial")
        svm_train(full_train_att, full_train_labels, initial_C, pi=pi, model="radial", pca=5)

        # ### -------z-score----------- ###
        svm_train(z_data, full_train_labels, initial_C, pi=pi)
        svm_train(z_data, full_train_labels, initial_C, pi=pi, model="polynomial")
        svm_train(z_data, full_train_labels, initial_C, pi=pi, model="polynomial", pca=5, z=1)
        svm_train(z_data, full_train_labels, initial_C, pi=pi, model="radial")
        svm_train(z_data, full_train_labels, initial_C, pi=pi, model="radial", pca=5, z=1)

        ### -------Gamma search----------- ###
        print("Gamma Search for hyperparameter")
        svm_gamma_train(full_train_att, full_train_labels, 0.1, model="radial", gamma=gamma)
        svm_gamma_train(z_data, full_train_labels, 0.1, model="radial", gamma=gamma)

        ## -------b and c search----------- ###
        print("b and c search for hyperparameter")
        full_dataset_C = np.logspace(-5, -1, 8)
        svm_b_c_train(full_train_att, full_train_labels, full_dataset_C, [2, 3], [0,1], model="polynomial")
        z_dataset_C = np.logspace(-2, 1, 8)
        svm_b_c_train(z_data, full_train_labels, z_dataset_C, [2, 3], [0,1], model="polynomial")

    if gmm_flag:
        models = ["GMM", "Tied", "Naive"]
        labels = 1
        pca = 0
        for model in models:
            print(f"Training {model}...")
            ## Can be a little slow, if want a faster result, delete last headers value
            headers = [
            f"{model}:[0.1, 0, 0.01]",
            f"{model}:[0.1, 1, 0.01]",
            f"{model}:[0.1, 2, 0.01]",
            f"{model}:[0.1, 3, 0.01]",
            f"{model}:[0.1, 4, 0.01]",
            ]
            
            ## Training without PCA
            [raw_values, S_raw] = call_GMM(full_train_att, full_train_labels, priorProb, pi=pi, models=headers, m=pca)
            [z_values, S_z] = call_GMM(z_data, full_train_labels, priorProb, pi=pi, models=headers, m=pca)
            graph_data(raw_values, z_values, f"{model} with PCA" if pca else f"model", p_labels=labels)

            ## Training with PCA
            pca = 5
            [raw_values, S_raw] = call_GMM(full_train_att, full_train_labels, priorProb, pi=pi, models=headers, m=pca)
            [z_values, S_z] = call_GMM(z_data, full_train_labels, priorProb, pi=pi, models=headers, m=pca)
            graph_data(raw_values, z_values, f"{model} with PCA" if pca else f"model", p_labels=labels)
    
    if score_cal:
        [SQuad, labels_quad, minDCF_quad] = k_fold(k, full_train_att, full_train_labels, priorProb, model="regression", pi=pi, pit=0.5, quadratic=1, score_calibration_flag=1)
        print("Quadratic MinDCF: ", (minDCF_quad[0.5] + minDCF_quad[0.1])/2)
        [SRadial, labels_radial, minDCF_radial] = k_fold(k, full_train_att, full_train_labels, priorProb, model="radial", pi=pi, pit=0.5, gamma=0.01, score_calibration_flag=1)
        print("Radial MinDCF: ", (minDCF_radial[0.5] + minDCF_radial[0.1])/2)
        [Spoly, labels, minDCF] = k_fold(k, z_data, full_train_labels, priorProb, model="polynomial", pi=pi, pit=0.5, score_calibration_flag=1, C=0.1)
        print("Polynomial MinDCF: ", (minDCF[0.5] + minDCF[0.1])/2)
        [SGMM, labels_gmm, minDCF_gmm] = k_fold(k, z_data, full_train_labels, priorProb, model="GMM", pi=pi, pit=0.5, niter=2, alpha=0.1, psi=0.01, score_calibration_flag=1)
        print("GMM MinDCF: ", (minDCF_gmm[0.5] + minDCF_gmm[0.1])/2)

        BayesErrorPlot(SQuad, labels_quad, Cfn, Cfp, model="quadratic")
        BayesErrorPlot(SRadial, labels_radial, Cfn, Cfp, model="Radial")
        BayesErrorPlot(Spoly, labels, Cfn, Cfp, model="polynomial")
        BayesErrorPlot(SGMM, labels_gmm, Cfn, Cfp, model="GMM")

        # Calibrate the score
        pi = [0.1, 0.5]
        labels_quad = np.array(labels_quad)
        labels_radial = np.array(labels_radial)
        labels = np.array(labels)
        labels_gmm = np.array(labels_gmm)

        scQuad_cal = score_calibration(5, SQuad, labels_quad, 0.00001)
        scQuad_cal = np.array(scQuad_cal)
        labels_quad = labels_quad[:-1]
        print("Quadratic MinDCF Calibrated: ", (minCostBayes(scQuad_cal, labels_quad, pi[1], Cfn, Cfp)[0] + minCostBayes(scQuad_cal, labels_quad, pi[0], Cfn, Cfp)[0])/2)
        scRadial_cal = score_calibration(5, SRadial, labels_radial, 0.00001)
        scRadial_cal = np.array(scRadial_cal)
        labels_radial = labels_radial[:-1]
        print("MinDCF Naive Calibrated: ", (minCostBayes(scRadial_cal, labels_radial, pi[1], Cfn, Cfp)[0] + minCostBayes(scRadial_cal, labels_radial, pi[0], Cfn, Cfp)[0])/2)
        scPoly_cal = score_calibration(5, Spoly, labels, 0.00001)
        scPoly_cal = np.array(scPoly_cal)
        labels = labels[:-1]
        print("Polynomial MinDCF Calibrated: ", (minCostBayes(scPoly_cal, labels, pi[1], Cfn, Cfp)[0] + minCostBayes(scPoly_cal, labels, pi[0], Cfn, Cfp)[0])/2)
        scGMM_cal = score_calibration(5, SGMM, labels_gmm, 0.00001)
        scGMM_cal = np.array(scGMM_cal)
        labels_gmm = labels_gmm[:-1]
        print("GMM MinDCF Calibrated: ", (minCostBayes(scGMM_cal, labels_gmm, pi[1], Cfn, Cfp)[0] + minCostBayes(scGMM_cal, labels_gmm, pi[0], Cfn, Cfp)[0])/2)

        BayesErrorPlot(scQuad_cal, labels_quad, Cfn, Cfp, model="quadratic Calibrated")
        BayesErrorPlot(scRadial_cal, labels_radial, Cfn, Cfp, model="Radial Calibrated")
        BayesErrorPlot(scPoly_cal, labels, Cfn, Cfp, model="polynomial Calibrated")
        BayesErrorPlot(scGMM_cal, labels_gmm, Cfn, Cfp, model="GMM Calibrated")

    if evaluation_flag:
        minDCffull_list=[]
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

        ## Quadratic Regression
        print("Training QR...")
        [_, _, minDCF, w, b] = k_fold(
            k,
            full_train_att,
            full_train_labels,
            priorProb,
            "regression",
            pi=pi,
            pit=pit,
            quadratic=1,
            l=l,
            final=1,
        )

        # ## Test Quadratic Regression
        print("Testing QR...")
        scr_quadra = calculate_logreg(full_test_att, w, b, 1)
        for p in pi:
            (minDCF, FPRQR, FLRQR, _) = minCostBayes(scr_quadra, full_test_labels, p, Cfn, Cfp)
            minDCffull_list.append(minDCF)
        print("Quadratic Regression MinDCF: ", minDCffull_list, "Mean: ", np.mean(minDCffull_list))
        BayesErrorPlot(scr_quadra, full_test_labels, Cfn, Cfp, "Quadratic Regression")

        # Calculate Radial SVM
        print("Training RSVM...")
        [SRad,  _, attribute_tra_rbm, labels_tra_rbm, x] = k_fold(
            k,
            full_train_att,
            full_train_labels,
            priorProb,
            "radial",
            pi=[0.1, 0.5],
            C=C,
            gamma=gamma,
            final=1,
        )

        _, w, b = score_calibration(5, SRad, labels_tra_rbm, 0.00001, testing=True)

        # Test Radial SVM
        print("Testing RSVM...")
        zi = 2 * labels_tra_rbm - 1
        scr_rad = calculate_svm(x, zi, attribute_tra_rbm, full_test_att, {"model":"radial", "gamma":gamma, "epsilon":0.1})
        scr_rad_cal = np.dot(w.T, vrow(scr_rad)) + b
        for p in pi:
            (minDCF, FPR_rad, FLR_rad, _) = minCostBayes(scr_rad_cal, full_test_labels, p, Cfn, Cfp)
            minDCffull_list.append(minDCF)
        print("Radial SVM MinDCF: ", minDCffull_list[-2:], "Mean: ", np.mean(minDCffull_list[-2:]))
        BayesErrorPlot(scr_rad_cal, full_test_labels, Cfn, Cfp, "Radial SVM Calibrated")

        ## Calculate Polynomial SVM
        print("Training PSVM...")
        [Spoly,  minDCF, attribute_tra_poly, labels_tra_poly, x] = k_fold(
            k,
            z_data,
            full_train_labels,
            priorProb,
            "polynomial",
            pi=[0.1, 0.5],
            C=C,
            dim=dim,
            c=c,
            final=1,
        )
        _, w, b = score_calibration(5, Spoly, labels_tra_poly, 0.00001, testing=True)

        ## Test Polynomial SVM
        print("Testing PSVM...")
        zi = 2 * labels_tra_poly - 1
        scr_poly = calculate_svm(x, zi, attribute_tra_poly, z_test, {"model":"polynomial", "dim":dim, "c":c, "epsilon":0.1})
        scr_poly_cal = np.dot(w.T, vrow(scr_poly)) + b
        for p in pi:
            (minDCF, FPRPSV, FLRPSV, _) = minCostBayes(scr_poly_cal, full_test_labels, p, Cfn, Cfp)
            minDCffull_list.append(minDCF)
        print("Polynomial SVM MinDCF: ", minDCffull_list[-2:], "Mean: ", np.mean(minDCffull_list[-2:]))
        BayesErrorPlot(scr_poly_cal, full_test_labels, Cfn, Cfp, "Polynomial SVM")

        # Calculate GMM
        print("Training GMM...")
        [Sgmm, labels_gmm, minDCF, multi_mu, cov, class_w] = k_fold(
            k,
            z_data,
            full_train_labels,
            priorProb,
            "GMM",
            pi=pi,
            alpha=alpha,
            psi=psi,
            niter=2,
            final=1,
        )

        ## Test GMM
        print("Testing GMM...")
        scr_gmm = calculate_gmm(z_test, full_test_labels, multi_mu, cov, class_w)
        for p in pi:
            (minDCF, FPRL, FNRL, _) = minCostBayes(scr_gmm, full_test_labels, p, Cfn, Cfp)
            minDCffull_list.append(minDCF)
        print("GMM MinDCF: ", minDCffull_list[-2:], "Mean: ", np.mean(minDCffull_list[-2:]))
        BayesErrorPlot(scr_gmm, full_test_labels, Cfn, Cfp, "GMM")

        # ROC Curve
        plt.figure()
        ROCcurve(FPRL, FNRL, model="GMM")
        ROCcurve(FPRQR, FLRQR, model="Quadratic Regression")
        ROCcurve(FPRPSV, FLRPSV, model="Polynomial SVM")
        ROCcurve(FPR_rad, FLR_rad, model="Radial SVM")
        plt.legend()
        plt.title("ROC Curve")
        plt.grid()
        plt.show()

        # DET Curve
        plt.figure()
        det_plot(FPRL, FNRL, color='b', model = "GMM",show=False)
        det_plot(FPRQR, FLRQR, color='r', model = "Quadratic Regression",show=False)
        det_plot(FPRPSV, FLRPSV, color='g', model = "Polynomial SVM",show=False)
        det_plot(FPR_rad, FLR_rad, color='y', model = "Radial SVM",show=False)
        plt.legend()
        plt.title("DET Curve")
        plt.show()
