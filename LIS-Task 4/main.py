import pandas as pd
import numpy as np
import csv
train = pd.read_hdf("train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")


print "Creating data\n"
# Grab the correct indices from the training data
X = train.ix[:,1:129].as_matrix()
y = train.ix[:, 0:1].as_matrix()

A = train_unlabeled.ix[:,0:128].as_matrix()

from sknn.mlp import Classifier, Layer

# This is the important stuff to adjust
print "Creating classifier\n"
nn = Classifier(
	layers=[Layer('Tanh', units=128), Layer('Sigmoid', units=128), Layer('Softmax', units=10)],
	learning_rate=.04,
	n_iter=85,
	batch_size = 10
)
"""
Uncomment to actually train whole data and write file
"""
outfile = open('output.csv','w') # change the file name
writer = csv.writer(outfile)
writer.writerow(['Id', 'y'])
print "About to fit\n"
nn.fit(X, y)
print "About to predict"
b = nn.predict(A)
nn.fit(A,b)
prediction = nn.predict(test.as_matrix())
print prediction

ids = test.ix[:, 0:1]
for i in range(prediction.shape[0]):
 	writer.writerow([i+30000, prediction[i][0]])

outfile.close()