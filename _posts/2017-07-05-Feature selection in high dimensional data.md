from sklearn.feature_selection import f_regression
import numpy as np
from sklearn import svm
from sklearn import linear_model
import svmcrossvalidate
from array import array

# Main 
#f = open("testdata1.txt")
f = open("testdata.txt")
mylist = f.readlines()
testdata = []
for i in range(0, len(mylist), 1):
	l = mylist[i].split()
	for j in range(0, len(l), 1):
		l[j] = float(l[j])
	#testdata.append(l)
	testdata.append(array('f',l))
f.close()

#print(testdata)

#f = open("traindata1.txt")
f = open("traindata.txt")
mylist = f.readlines()
train = []
for i in range(0, len(mylist), 1):
	l = mylist[i].split()
	for j in range(0, len(l), 1):
		l[j] = float(l[j])
	#train.append(l)
	train.append(array('f',l))
f.close()
#print(train)
#X is for train data
X = train

#f = open("trueclass1.txt")
f = open("trueclass.txt")
mylist = f.readlines()
trainlabels = []
for i in range(0, len(mylist), 1):
	l = mylist[i].split()
	for j in range(0, len(l), 1):
		l[j] = float(l[j])
	trainlabels.append(l[0])
f.close()

y = trainlabels
#print(trainlabels)

#f_output = f_regression(X,y)
f_output = f_regression(X, y, center=True)

#print(f_output[0])
#print(f_output[1])

cols = len(X[0])
indices = []
for i in range(0, cols, 1):
        indices.append(i)
fscores = f_output[0]
fscores_dict = {}
for i in range(0, len(f_output[0]), 1):
        fscores_dict[i] = fscores[i]

sorted_indices = sorted(indices, key=fscores_dict.__getitem__, reverse=True)

#print(sorted_indices)
print(sorted_indices[:15])

# Reduce both traindata and testdata to top 15 ranked features


newtestdata= []
newtrain = []

rows = len(testdata)
cols = len(testdata[0])

print("**testdata**")
print(rows)
print(cols)

for i in range(0, rows, 1):
        l1 = []
        for j in range(0, cols, 1):
                if (j in sorted_indices[:15]):
                        l1.append(testdata[i][j])
        newtestdata.append(l1)

rows = len(train)
cols = len(train[0])

print("**traindata**")
print(rows)
print(cols)

for i in range(0, rows, 1):
        l2 = []
        for j in range(0, cols, 1):
                if (j in sorted_indices[:15]):
                        l2.append(train[i][j])        
        newtrain.append(l2)

#print(newtestdata)               
#print(newtrain)


##### Cross-validated linear SVM #####

[bestC,besterror] = svmcrossvalidate.getbestC(newtrain,trainlabels)

print("Best C = ", bestC)
print("Best cross validation error = ", besterror)

# Predict labels of test data

clf = svm.LinearSVC(C=bestC, max_iter=100000)
clf.fit(train,trainlabels)
prediction = clf.predict(testdata)
   
f = open("testlabel_prediction.txt", 'w')
for i in range(0, len(prediction), 1):
        #print("Predict test label:", int(prediction[i]))
        f.write(str(int(prediction[i]))+ " " + str(i) + "\n")
f.close()        

