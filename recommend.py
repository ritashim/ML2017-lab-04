import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_table('u1.base',header=None,encoding='gb2312',delim_whitespace=True,index_col=None)
test_df = pd.read_table('u1.test',header=None,encoding='gb2312',delim_whitespace=True,index_col=None)

train_df = train_df.drop(3, axis=1)
train_df.columns = ['user_id','item_id','rating']
test_df = test_df.drop(3, axis=1)
test_df.columns = ['user_id','item_id','rating']

train = train_df.values
test = test_df.values

all_df = pd.concat([train_df, test_df],ignore_index=True)
n_users = len(set(all_df['user_id']))
n_items = len(set(all_df['item_id']))
print('number of users:',n_users)
print('number of items:',n_items)

train_set = pd.concat([test_df, train_df],ignore_index=True)
train_set.loc[range(len(test_df)),'rating'] = 0
train_R = train_set.pivot(index='user_id',columns='item_id',values='rating').fillna(0)
train_R = train_R.values

test_set = pd.concat([train_df, test_df],ignore_index=True)
test_set.loc[range(len(train_df)),'rating'] = 0
test_R = test_set.pivot(index='user_id',columns='item_id',values='rating').fillna(0)
test_R = test_R.values

def loss(R, P, Q, Lambda):
    error = 0
    for row in R:
        user_id = row[0]-1
        item_id = row[1]-1
        rating = row[2]
        error += (rating - np.dot(P[user_id], Q[item_id]))**2
    return error / R.shape[0] + Lambda * (np.sum(P * P) + np.sum(Q * Q))


K = 10
reg = 0.01
learning_rate = 0.005
iteration_num = 20000
Lambda=0.0001
batch_number=100
#initialization
P = np.random.rand(n_users,K)
Q = np.random.rand(n_items,K)



SGD_train_loss_history = []
SGD_test_loss_history = []
for i in range(iteration_num):
    Gradient_user = np.zeros(10)
    Gradient_item = np.zeros(10)
    for j in range(batch_number):
        index = np.random.randint(len(train))
        row = train[index]
        user_id = row[0] - 1
        item_id = row[1] - 1
        rating = row[2]
        # calculate gradient
        prediction_error = rating - np.dot(P[user_id], Q[item_id])
        Gradient_user += -prediction_error*Q[item_id]+Lambda*P[user_id]
        Gradient_item += -prediction_error*P[user_id]+Lambda*Q[item_id]
    Gradient_user /= batch_number
    Gradient_item /= batch_number
    # update P&Q
    P[user_id] = P[user_id] - learning_rate * Gradient_user
    Q[item_id] = Q[item_id] - learning_rate * Gradient_item
    print('\riteration: '+str(i)+"/" + str(iteration_num), end=" ")
    if(i%10==0):
        SGD_train_loss_history.append(loss(train,P,Q,Lambda))
        SGD_test_loss_history.append(loss(test,P,Q,Lambda))


plt.title('SGD')
plt.xlabel('iteration number')
plt.ylabel('loss')
plt.plot(range(len(SGD_train_loss_history)), SGD_train_loss_history, label='SGD train loss')
plt.plot(range(len(SGD_test_loss_history)), SGD_test_loss_history, label='SGD test loss')
plt.legend()
plt.grid()
plt.show()