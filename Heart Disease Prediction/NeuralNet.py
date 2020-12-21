import numpy as np
import matplotlib.pyplot as plt

class NeuralNet():
  def __init__(self, layers=[13, 8, 1], learning_rate=0.001, iterations=100):
    self.params = {}
    self.learning_rate = learning_rate
    self.iterations = iterations
    self.loss = []
    self.sample_size = None
    self.layers = layers
    self.X = None
    self.y = None
  
  def init_weights(self):
    np.random.seed(1)
    self.params['w1'] = np.random.randn(self.layers[0], self.layers[1])
    self.params['b1'] = np.random.randn(self.layers[1],)
    self.params['w2'] = np.random.randn(self.layers[1], self.layers[2])
    self.params['b2'] = np.random.randn(self.layers[2],)
  
  def relu(self, Z):
    return np.maximum(0,Z)
  
  def sigmoid(self, Z):
    return 1.0/(1.0 + np.exp(-Z))
  
  def entropy_loss(self, y, y_hat):
    nsample = len(y)
    loss = -1/nsample * (np.sum(np.multiply(np.log(y_hat), y) + np.multiply((1 -y), np.log(1 - y_hat))))
    return loss
  
  def forward_propagation(self):
    Z1 = self.X.dot(self.params['w1']) + self.params['b1']
    A1 = self.relu(Z1)
    Z2 = A1.dot(self.params['w2']) + self.params['b2']
    y_hat = self.sigmoid(Z2)
    loss = self.entropy_loss(self.y, y_hat)
    
    self.params['z1'] = Z1
    self.params['z2'] = Z2
    self.params['a1'] = A1
    
    return y_hat, loss
  
  def back_propagation(self, y_hat):
    def dRelu(x):
      x[x<=0] = 0
      x[x>0] = 1
      return x
    
    dl_yhat = -(np.divide(self.y, y_hat) - np.divide((1-self.y), (1 - y_hat)))
    dl_sig = y_hat * (1 - y_hat)
    dl_z2 = dl_yhat * dl_sig
    dl_a1 = dl_z2.dot(self.params['w2'].T)
    dl_w2 = self.params['a1'].T.dot(dl_z2)
    dl_b2 = np.sum(dl_z2, axis=0)
    dl_z1 = dl_a1 * dRelu(self.params['z1'])
    dl_w1 = self.X.T.dot(dl_z1)
    dl_b1 = np.sum(dl_z1, axis=0)
    
    self.params['w1'] = self.params['w1'] - self.learning_rate * dl_w1
    self.params['w2'] = self.params['w2'] - self.learning_rate * dl_w2
    self.params['b1'] = self.params['b1'] - self.learning_rate * dl_b1
    self.params['b2'] = self.params['b2'] - self.learning_rate * dl_b2
    
  def fit(self, X, y):
    self.X = X
    self.y = y
    
    self.init_weights()
    
    for i in range(self.iterations):
      y_hat, loss = self.forward_propagation()
      self.back_propagation(y_hat)
      self.loss.append(loss)
    
  def predict(self, X):
    z1 = X.dot(self.params['w1']) + self.params['b1']
    a1 = self.relu(z1)
    z2 = a1.dot(self.params['w2']) + self.params['b2']
    pred = self.sigmoid(z2)
    return np.round(pred)
  
  def acc(self, y, y_hat):
    acc = int(sum(y == y_hat) / len(y) * 100)
    return acc
  
  def plot_loss(self):
    plt.plot(self.loss)
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.title('Loss curve for training')
    plt.show()