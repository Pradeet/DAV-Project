from scipy import stats
import scipy.io as spio
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scikits.bootstrap as bootstrap

def pdu(data):
    mean = np.mean(data)
    pdu = np.sum(data<mean,axis=1)
    return pdu/len(data[0])*100

# def bootstrap(data):
#     samples = []
#     B = 10000
#     N = len(data)
#     for i in range(B):
#         c_sample = []
#         for i in range(N):
#             c_sample.append(data[np.random.randint(0, N)])
#         samples.append(c_sample)
#     means = np.mean(samples, axis=1)
#     plt.hist(means)
#     plt.show()
#     return np.mean(means), np.var(means)

def linearRegression(X, Y):
    a, b = np.polyfit(X, Y, deg=1)
    return a * X + b

data = spio.loadmat('data_CS306.mat')

data100 = data['data_100']
data144 = data['data_144']

mean100, var100 = np.mean(data100, axis=1), np.var(data100, axis=1)
mean144, var144 = np.mean(data144, axis=1), np.var(data144, axis=1)

pdu100 = pdu(data100)
pdu144 = pdu(data144)

pdu100_pd = linearRegression(mean100, pdu100)
pdu144_pd = linearRegression(mean144, pdu144)

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

# b_mean100, b_var100 = bootstrap(pdu100)
# b_mean144, b_var144 = bootstrap(pdu144)
#
# print("Mean and variance of pdu100: ", np.mean(pdu100), "  ", np.var(pdu100))
# print("Mean and variance of b_pdu100: ", b_mean100, "  ", b_var100)
#
# print("\nMean and variance of pdu144: ", np.mean(pdu144), "  ", np.var(pdu144))
# print("Mean and variance of b_pdu144: ", b_mean144, "  ", b_var144)

# CIs = bootstrap.ci(data=pdu100_pd, statfunction=np.mean, alpha=0.05)
# print(CIs)

# print("Mean of calculated PDU of HDR videos:", np.mean(pdu100))
# print("Variance of calculated PDU of HDR videos", np.var(pdu100))
# print("Variance of predicted PDU using Linear Regression of HDR videos", np.var(pdu100_pd))
# print("p-value of f-test between actual and predicted PDU of HDR videos", p100)
#
# print("\nMean of calculated PDU of Full HD videos:", np.mean(pdu144))
# print("Variance of calculated PDU of Full HD videos", np.var(pdu144))
# print("Variance of predicted PDU using Linear Regression of Full HD videos", np.var(pdu144_pd))
# print("p-value of f-test between actual and predicted PDU of Full HD videos", p144)
