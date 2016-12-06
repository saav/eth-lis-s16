import numpy as np
import csv
from sklearn import svm
from sklearn import grid_search
from sklearn.svm import SVC
#clf = svm.NuSVC(kernel='rbf', decision_function_shape='ovr',gamma=.0057, probability=True)
#clf = KNeighborsClassifier(n_neighbors=8, weights='distance', algorithm='ball_tree')
clf = SVC(kernel='rbf', gamma=.0019,C=1.3)
trainfile = open('train.csv', 'r')
testfile = open('test.csv', 'r')
outfile = open('out4.csv','w')

reader = csv.reader(trainfile)
writer = csv.writer(outfile)
y = []
X = []
for row in reader:
	if reader.line_num != 1:
		X.append(map(float, row[2:]))
		y.append(float(row[1]))
X = np.array(X)
y = np.array(y)

"""
GRID SEARCH
"""

"""
TRAINING
"""
clf.fit(X, y)

"""
OUR TEST DATA TO BE OUTPUTTED
"""
X_test = []
reader_2 = csv.reader(testfile)
rows = []
for row in reader_2:
	if reader_2.line_num != 1:
		X_test.append(map(float, row[1:]))
		rows.append(row[0])

X_test = np.array(X_test)
writer.writerow(['Id', 'y'])
for i in range(len(X_test)):
	y = clf.predict([X_test[i]])[0]
	writer.writerow([int(rows[i]), y])

outfile.close()

"""
CROSS VALIDATION METRIC
"""
# train = 50 # The number to save for cross validation
# num_wrong = 0
# reps = (1000 / train)

# for i in range(reps):
# 	curr_x_train = np.delete(X, range(i*train, (i+1)*train), 0)
# 	curr_y_train = np.delete(y, range(i*train, (i+1) * train))
# 	clf.fit(curr_x_train, curr_y_train)
# 	curr_x_test = X[i*train:(i+1)*train, :]
# 	curr_y_test = y[i*train:(i+1)*train]
# 	#print curr_x_test
# 	y_pred = clf.predict(curr_x_test)
# 	#print y_pred
# 	for j in range(len(curr_y_test)):
# 		if y_pred[j] != curr_y_test[j]:
# 			num_wrong += 1
# 	#print num_wrong
# 	#raw_input()
# print num_wrong

# import code
# vars = globals().copy()
# vars.update(locals())
# shell = code.InteractiveConsole(vars)
# shell.interact()