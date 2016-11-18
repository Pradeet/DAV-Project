from scipy import stats
import scipy.io as spio
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

def pdu(data):
    mean = np.mean(data)
    pdu = np.sum(data<mean,axis=1)
    return pdu/len(data[0])*100


data = spio.loadmat('data_CS306.mat')

data100 = data['data_100']
data144 = data['data_144']

mean100 = np.mean(data100, axis=1)
mean144 = np.mean(data144, axis=1)

pdu100 = pdu(data100)
pdu144 = pdu(data144)

model = GaussianProcessRegressor()

model.fit(mean100.reshape(-1, 1), pdu100)
pdu100_pd = model.predict(mean100.reshape(-1,1))

model.fit(mean144.reshape(-1, 1), pdu144)
pdu144_pd = model.predict(mean144.reshape(-1,1))

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
print("Variance of predicted PDU using Gaussian Regression of HDR videos", np.var(pdu100_pd))
print("p-value of f-test between actual and predicted PDU of HDR videos", p100)

print("\nMean of calculated PDU of Full HD videos:", np.mean(pdu144))
print("Variance of calculated PDU of Full HD videos", np.var(pdu144))
print("Variance of predicted PDU using Gaussian Regression of Full HD videos", np.var(pdu144_pd))
print("p-value of f-test between actual and predicted PDU of Full HD videos", p144)
