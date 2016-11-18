from scipy import stats
import scipy.io as spio
import numpy as np
from scipy.optimize import minimize

def pdu(data):
    mean = np.mean(data)
    pdu = np.sum(data<mean,axis=1)
    return pdu/len(data[0])*100

def LogisticRegression(data, para):
    return para[0]*(0.5 - 1 / (1 + np.exp(para[1]*(para[2] - data)))) + para[3]

def mean_sqr_100(para):
    return np.sum((pdu100 - LogisticRegression(mean100, para))**2)

def mean_sqr_144(para):
    return np.sum((pdu144 - LogisticRegression(mean144, para))**2)

data = spio.loadmat('data_CS306.mat')

data100 = data['data_100']
data144 = data['data_144']

mean100 = np.mean(data100, axis=1)
mean144 = np.mean(data144, axis=1)

pdu100 = pdu(data100)
pdu144 = pdu(data144)

pdu100_pd = LogisticRegression(mean100, minimize(mean_sqr_100, [9, 2, 3, 2]).x)
pdu144_pd = LogisticRegression(mean144, minimize(mean_sqr_144, [10, 2, 1, 8]).x)

def f_score(a, b):
    v_a = np.var(a)
    v_b = np.var(b)
    df_a = len(a) - 1
    df_b = len(b) - 1
    if v_a > v_b:
        f = v_a/v_b
    else:
        f = v_b/v_a
    return stats.f.cdf(f, df_a, df_b)

p100 = f_score(pdu100_pd, pdu100)
p144 = f_score(pdu144_pd, pdu144)

print("Mean of calculated PDU of HDR videos:", np.mean(pdu100))
print("Variance of calculated PDU of HDR videos", np.var(pdu100))
print("Variance of predicted PDU using Logistic Regression of HDR videos", np.var(pdu100_pd))
print("p-value of f-test between actual and predicted PDU of HDR videos", p100)

print("\nMean of calculated PDU of Full HD videos:", np.mean(pdu144))
print("Variance of calculated PDU of Full HD videos", np.var(pdu144))
print("Variance of predicted PDU using Logistic Regression of Full HD videos", np.var(pdu144_pd))
print("p-value of f-test between actual and predicted PDU of Full HD videos", p144)
