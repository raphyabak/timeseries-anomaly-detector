import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


def main():
    st.title("Time Series Anomaly")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Time Series Anomaly Detector </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
if __name__=='__main__':
    main()

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:

# dataframe = pd.read_csv('TimeSeries.csv')

 dataframe = pd.read_csv(uploaded_file)
 raw_data = dataframe.values
 columns=dataframe.shape[1] - 1

# The last element contains the labels
 labels = raw_data[:, -1]

# The other data points are the electrocadriogram data
 data = raw_data[:, 0:-1]

 train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21
 )

 min_val = tf.reduce_min(train_data)
 max_val = tf.reduce_max(train_data)

 train_data = (train_data - min_val) / (max_val - min_val)
 test_data = (test_data - min_val) / (max_val - min_val)

 train_data = tf.cast(train_data, tf.float32)
 test_data = tf.cast(test_data, tf.float32)

 train_labels = train_labels.astype(bool)
 test_labels = test_labels.astype(bool)

 normal_train_data = train_data[train_labels]
 normal_test_data = test_data[test_labels]

 anomalous_train_data = train_data[~train_labels]
 anomalous_test_data = test_data[~test_labels]

 class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])
    
    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(columns, activation="sigmoid")])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

 autoencoder = AnomalyDetector()

 autoencoder.compile(optimizer='adam', loss='mae')

 history = autoencoder.fit(normal_train_data, normal_train_data, 
          epochs=20, 
          batch_size=512,
          validation_data=(test_data, test_data),
          shuffle=True)

 encoded_data = autoencoder.encoder(normal_test_data).numpy()
 decoded_data = autoencoder.decoder(encoded_data).numpy()

 encoded_data = autoencoder.encoder(anomalous_test_data).numpy()
 decoded_data = autoencoder.decoder(encoded_data).numpy()

 reconstructions = autoencoder.predict(normal_train_data)
 train_loss = tf.keras.losses.mae(reconstructions, normal_train_data) 

 threshold = np.mean(train_loss) + np.std(train_loss)
 print("Threshold: ", threshold)

 reconstructions = autoencoder.predict(anomalous_test_data)
 test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)

 def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

 def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))
#   if st.button("Detect Anomaly"):
  st.success("Anomaly = {}".format(accuracy_score(labels, predictions)))
#   st.line_chart(dataframe.head()) 

 preds = predict(autoencoder, test_data, threshold)
 print_stats(preds, test_labels)


st.button("Detect Anomaly")

