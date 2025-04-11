import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pprint import pprint


#load and normalize the data
df = pd.read_csv('ex_1.csv')
x= np.array([1,2,3])
print(x.max())

df['time'] = (df['time'] -df['time'] .mean())/(df['time'].std())
x_feature = np.array(df['time'])
df['el_power'] = (df['el_power'] -df['el_power'] .mean())/(df['el_power'].std())
y_targets = np.array(df['el_power'])

#for show the method of L.regression well
x_feature = x_feature[0:100]
y_targets=y_targets[0:100]

#initialize Parameters
w=.6862611738384387
b=1.002336064449456

#hyperparameters 
m = len(x_feature)  #num of learining examples 
alpha = .01         #learning rate
epochs=1000       #how many time i will make batch gradient descent

#functions to computer cost_value 
def cost_value(m,y_pred,y_target):
    cost_haf= 0
    cost_haf2=[]
    for i in range(m):
        cost_haf = (y_pred[i]-y_target[i])**2
        cost_haf2.append(cost_haf)
    cost= sum(cost_haf2)
    final_cost=(1/(2*m))*cost
    return final_cost


#Gradient descent loop 
#1- I use w,b then calculate y_pred at them, then get cost value at this w,b 
#2-we use Gradient descent to choose ower new comined w,b and reset for next epoch
#3-we repete this for all epochs we have wish to find smallest cost error in some combination 
cost_history = []
for i in range(epochs):   #we use batch method to do this loop!
    y_pred = x_feature *w + b  #this can be done as we make x_feature array , and array has element wise operations
    cost_history.append(cost_value(m,y_pred,y_targets))

    dw=(1 / m) * np.sum((y_pred - y_targets) * x_feature)
    db=(1 / m) * np.sum(y_pred - y_targets)
    w=w-alpha*(dw)
    b=b-alpha*(db)
    if np.array(cost_history).min() <= .0001:  #choose any error you accept 
        break

#for erro at break case 
epochs = len(cost_history)
#results
print("Final w (slope):", w)
print("Final b (intercept):", b)
print("the w&b cost function values :",np.array(cost_history).min() )

y_final = w*x_feature+b
plt.figure()
plt.scatter(x_feature,y_targets,color='r',label="actual data")
plt.scatter(x_feature,y_final,color='k',label="module data")
plt.xlabel("x_feature_normalized")
plt.ylabel('y_targets_normalized')
plt.title("Linear regression for w & b")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(range(epochs), cost_history)
plt.xlabel('Epoch')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function Convergence')
plt.grid(True)
plt.show()