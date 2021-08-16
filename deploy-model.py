from model import *
from utility.NN_utils import *

'''RUN this script "deploy_model.py" to train the neural network!
   After the training, the model architecture and the weights are saved!'''

## Loading training data ##
X,y = load_training_data()
print(np.shape(X),np.shape(y))
n = len(y)
val_i = n//10
## split into train&validation sets ##
X_train = X[0:n-val_i,:]
y_train = y[:n-val_i]
X_val =  X[-val_i:,:]
y_val = y[-val_i:]
shape_X = np.shape(X_train)[1]
shape_y = np.shape(y_train)

## Initialize model and start training ##
model = NN()
chess_model = model.build_model(shape_X)
model.train(chess_model, X_train, y_train, X_val, y_val, 128)