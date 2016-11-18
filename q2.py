from scipy import stats
import scipy.io as spio
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
import scikits.bootstrap as bootstrap
from prettytable import PrettyTable as PT

def pdu(data):
    mean = np.mean(data)
    pdu = np.sum(data<mean,axis=1)
    return pdu/len(data[0])*100

def f_score(a, b):
    v_a = np.var(a)
    v_b = np.var(b)
    df_a = len(a) - 1
    df_b = len(b) - 1
    if v_a > v_b:
        f = v_a/v_b
    else:
        f = v_b/v_a
    return f, stats.f.cdf(f, df_a, df_b)

def linearRegression(X, Y):
    a, b = np.polyfit(X, Y, deg=1)
    return a * X + b

def LogisticRegression(data, para):
    return para[0]*(0.5 - 1 / (1 + np.exp(para[1]*(para[2] - data)))) + para[3]

def mean_sqr_100(para):
    return np.sum((pdu100 - LogisticRegression(mean100, para))**2)

def mean_sqr_144(para):
    return np.sum((pdu144 - LogisticRegression(mean144, para))**2)

def GaussianRegression(model, X, Y):
    model.fit(X.reshape(-1, 1), Y)
    return model.predict(X.reshape(-1, 1))

data = spio.loadmat('data_CS306.mat')

data100 = data['data_100']
data144 = data['data_144']

mean100, var100 = np.mean(data100, axis=1), np.var(data100, axis=1)
mean144, var144 = np.mean(data144, axis=1), np.var(data144, axis=1)

pdu100 = pdu(data100)
pdu144 = pdu(data144)

### Linear Regression ###

pdu100_pd_li = linearRegression(mean100, pdu100)
pdu144_pd_li = linearRegression(mean144, pdu144)

f100_li, p100_li = f_score(pdu100_pd_li, pdu100)
f144_li, p144_li = f_score(pdu144_pd_li, pdu144)

ci100_mean_l_li, ci100_mean_h_li = bootstrap.ci(data=pdu100_pd_li, statfunction=np.mean, alpha=0.05)
ci100_var_l_li, ci100_var_h_li = bootstrap.ci(data=pdu100_pd_li, statfunction=np.var, alpha=0.05)

ci144_mean_li, ci144_mean_hi = bootstrap.ci(data=pdu144_pd_li, statfunction=np.mean, alpha=0.05)
ci144_var_li, ci144_var_hi = bootstrap.ci(data=pdu144_pd_li, statfunction=np.var, alpha=0.05)

### Logistic Regression ###

pdu100_pd_lo = LogisticRegression(mean100, minimize(mean_sqr_100, [9, 2, 3, 2]).x)
pdu144_pd_lo = LogisticRegression(mean144, minimize(mean_sqr_144, [10, 2, 1, 8]).x)

f100_lo, p100_lo = f_score(pdu100_pd_lo, pdu100)
f144_lo, p144_lo = f_score(pdu144_pd_lo, pdu144)

ci100_mean_lo, ci100_mean_hi = bootstrap.ci(data=pdu100_pd_lo, statfunction=np.mean, alpha=0.05)
ci100_var_lo, ci100_var_hi = bootstrap.ci(data=pdu100_pd_lo, statfunction=np.var, alpha=0.05)

ci144_mean_lo, ci144_mean_hi = bootstrap.ci(data=pdu144_pd_lo, statfunction=np.mean, alpha=0.05)
ci144_var_lo, ci144_var_hi = bootstrap.ci(data=pdu144_pd_lo, statfunction=np.var, alpha=0.05)

### Gaussian Regressian ###

model = GaussianProcessRegressor()

pdu100_pd_ga = GaussianRegression(model, mean100, pdu100)
pdu144_pd_ga = GaussianRegression(model, mean144, pdu144)

f100_ga, p100_ga = f_score(pdu100_pd_ga, pdu100)
f144_ga, p144_ga = f_score(pdu144_pd_ga, pdu144)

ci100_mean_ga, ci100_mean_hi = bootstrap.ci(data=pdu100_pd_ga, statfunction=np.mean, alpha=0.05)
ci100_var_ga, ci100_var_hi = bootstrap.ci(data=pdu100_pd_ga, statfunction=np.var, alpha=0.05)

ci144_mean_ga, ci144_mean_hi = bootstrap.ci(data=pdu144_pd_ga, statfunction=np.mean, alpha=0.05)
ci144_var_ga, ci144_var_hi = bootstrap.ci(data=pdu144_pd_ga, statfunction=np.var, alpha=0.05)

### Mann Whitney U Test ###

man100_li_lo = stats.mannwhitneyu(pdu100_pd_li, pdu100_pd_lo, use_continuity=True)
man100_lo_ga = stats.mannwhitneyu(pdu100_pd_lo, pdu100_pd_ga, use_continuity=True)
man100_ga_li = stats.mannwhitneyu(pdu100_pd_ga, pdu100_pd_li, use_continuity=True)
print(man100_li_lo)
print(man100_lo_ga)
print(man100_ga_li)

### Analysis ###

print('\n'*2)
table = PT(['Model', 'HDR', 'Full HD'])
table.add_row(['', '(f-score, p-value)', '(f-score, p-value)'])
table.add_row(['Linear Regression', (f100_li, p100_li), (f144_li, p144_li)])
table.add_row(['Logistic Regression', (f100_lo, p100_lo), (f144_lo, p144_lo)])
table.add_row(['Gaussian Regression', (f100_ga, p100_ga), (f144_ga, p144_ga)])
print(table)

print("\n"*2)

table = PT(['Variance of Predicted PDU', 'HDR', 'Full HD'])
table.add_row(['Original', np.var(pdu100), np.var(pdu144)])
table.add_row(['Linear', np.var(pdu100_pd_li), np.var(pdu144_pd_li)])
table.add_row(['Logistic', np.var(pdu100_pd_lo), np.var(pdu144_pd_lo)])
table.add_row(['Gaussian', np.var(pdu100_pd_ga), np.var(pdu144_pd_ga)])
print(table)

print("\n", "-"*15, "Comparision between regressions for HDR", "-"*15)

print("Linear   - Logistic : ", f_score(pdu100_pd_li, pdu100_pd_lo))
print("Logistic - Gaussian : ", f_score(pdu100_pd_lo, pdu100_pd_ga))
print("Gaussian - Linear   : ", f_score(pdu100_pd_ga, pdu100_pd_li))

print("\n", "-"*15, "Bootstrapping")
