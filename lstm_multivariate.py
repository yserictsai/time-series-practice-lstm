import pickle
import sys
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from datetime import datetime, date
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	
    # time series input
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	
    # forecast 
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	
    # put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	
    # drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	
    return agg



train_path = '/data/examples/ibmxfet/training.csv'
test_path = '/data/examples/ibmxfet/testing.csv'
feat_path = 'Imp_feat_20.csv'

train_data = pd.read_csv(train_path, encoding='UTF-8')
test_data = pd.read_csv(test_path, encoding='UTF-8')
feat_data = pd.read_csv(feat_path, encoding='UTF-8')
feat_data = feat_data.drop(['Unnamed: 0', 'y', 'USER_ID', 'date'], axis=1)
feat_data = feat_data.fillna(0)

y = train_data['y']

train_data = train_data.drop(['y'], axis=1)
train_data = pd.concat([train_data, feat_data, y], axis=1)

test_data['y'] = np.nan



userid_ll = train_data["USER_ID"].unique().tolist()

ll_1 = userid_ll[:200]
ll_2 = userid_ll[200:400]
ll_3 = userid_ll[400:600]
ll_4 = userid_ll[600:800]
ll_5 = userid_ll[800:1200]
ll_6 = userid_ll[1200:1400]
ll_7 = userid_ll[1400:1600]
ll_8 = userid_ll[1600:1800]
ll_9 = userid_ll[1800:2000]
ll_10 = userid_ll[2000:2400]
ll_11 = userid_ll[2400:]
ll_t = userid_ll[1:2]
ll_key = sys.argv[1]

dd = {'ll_1': ll_1, 'll_2': ll_2, 'll_3': ll_3, 'll_4': ll_4, 'll_5': ll_5, 'll_6': ll_6, 'll_7': ll_7, 'll_8': ll_8, 'll_9': ll_9, 'll_10': ll_10, 'll_11': ll_11, 'll_t': ll_t}



y_predict = []


for i in dd[ll_key]:
    train_i = train_data[train_data["USER_ID"]==int(i)]
    test_i = test_data[test_data["USER_ID"]==int(i)]
    
    train_i.index = pd.to_datetime(train_i['date'])
    train_i = train_i.drop(['date','USER_ID'], axis=1)
    
    test_i.index = pd.to_datetime(test_i['date'])
    test_i = test_i.drop(['date','USER_ID'], axis=1)

    train_i = train_i.astype('float32')
    
    df = train_i.append(test_i)
    
    reframed = series_to_supervised(df, 1, 1)
    reframed.drop(reframed.columns[[range(22,43)]], axis=1, inplace=True)
    
    values = reframed.values
    train = values[:11, :]
    test = values[11:, :]

    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    
    # design network
    model = Sequential()
    model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=5, validation_data=(test_X, test_y), verbose=2, shuffle=False)


    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()
    
    
    yhat = model.predict(test_X)
    rmse = sqrt(mean_squared_error(test_y, yhat))
    print('Test RMSE: %.3f' % rmse)
    print('ID: %.1f' % i)
    
    X_predict = train_i.iloc[-1, :].values.reshape((1, 1, 22))
    y_t = model.predict(X_predict)
    y_predict.append(y_t.reshape(-1)[0])
    pickle.dump(y_predict, open(str(ll_key)+'_mul_y_predict.p', 'wb'))
    
submission = pd.DataFrame({'USER_ID': dd[ll_key], 'date': '2018-03-16', 'y': y_predict})
submission.to_csv(str(ll_key)+'_mul_prediction_lstm_1.csv', index=False)
print("done")