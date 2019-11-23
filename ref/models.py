
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso



def get_rf_model():
    
    return RandomForestRegressor(n_jobs=-1,
                              n_estimators=1000,
                              criterion='mae')


def get_lasso_model():
    
    steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Lasso(alpha=5.0, fit_intercept=True))
    ]

    lasso_pipe = Pipeline(steps)
    return lasso_pipe

def get_nn_model():

    # gives 25 or smth
    model = Sequential()
    model.add(Dense(200, input_shape=(80,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(400))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(450))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(150))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(150))
    model.add(Activation('relu'))
    
    # last layer    
    model.add(Dense(1))
    
    model.compile(optimizer='adam',
                      loss='mean_absolute_percentage_error', 
                      metrics=['mae','accuracy'])
    
    
    return model