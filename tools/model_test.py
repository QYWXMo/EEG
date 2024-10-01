
import numpy as np

def test_data(data, periods=1):
    y = data.values
    y = y[periods:]
    X_data = np.zeros((periods, len(y)))
    for i in range(periods):
        X = data.shift(periods=i+1)
        X.dropna(inplace=True)
        X = X.values
        if -periods+i+1 < 0:
            X = X[periods-i-1:]
        X = X.reshape(1, -1)
        X_data[i] = X[0]
    X_data = X_data.transpose()
    return X_data, y


def test_model(model, data, periods=20):
    predictions = []
    data = data.values
    features = data[:periods]
    features = features.reshape(1, -1)
    for i in range(len(data)-periods):
        pred = model.predict(features)
        predictions.append(pred)
        cur_features = np.delete(features[0], [0])
        cur_features = np.append(cur_features, pred)
        features[0] = cur_features

    return predictions



