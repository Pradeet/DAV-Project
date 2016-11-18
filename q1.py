from scipy import stats
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt

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

cor_cof100, p100 = stats.pearsonr(pdu100, mean100)
cor_cof144, p144 = stats.pearsonr(pdu144, mean144)

# plt.figure()
# plt.boxplot(pdu100)
# plt.show()

print("Corelation coefficient and p-value of HDR Videos: ", cor_cof100, p100)
print("Corelation coefficient and p-value of Full HD Videos: ", cor_cof144, p144)

fig, ax = plt.subplots()
a, b = np.polyfit(mean100, pdu100, deg=1)
ax.plot(mean100, a*mean100 + b, color='red')
ax.scatter(mean100, pdu100)
ax.set_title("HDR Video")
ax.set_xlabel("Mean Option Score (MOS)")
ax.set_ylabel("percentage dissatisfied users (PDU)")

fig, ax = plt.subplots()
a, b = np.polyfit(mean144, pdu144, deg=1)
ax.plot(mean144, a*mean144 + b, color='red')
ax.scatter(mean144, pdu144)
ax.set_title("HDR Video")
ax.set_xlabel("Mean Option Score (MOS)")
ax.set_ylabel("percentage dissatisfied users (PDU)")
plt.show()
