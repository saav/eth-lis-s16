import pandas as pd
import numpy as np
import csv
train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")


print "Creating data\n"
# Grab the correct indices from the training data
X = train.ix[:,1:101].as_matrix()
y = train.ix[:, 0:1].as_matrix()


from sknn.mlp import Classifier, Layer

# This is the important stuff to adjust
print "Creating classifier\n"
nn = Classifier(
	layers=[Layer('Tanh', units=100), Layer('Sigmoid', units = 25), Layer('Softmax', units=5)],
	learning_rate=.03,
	n_iter=73,
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
prediction = nn.predict(test.as_matrix())
print prediction

ids = test.ix[:, 0:1]
for i in range(prediction.shape[0]):
 	writer.writerow([i+45324, prediction[i][0]])

outfile.close()