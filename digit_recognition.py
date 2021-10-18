import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
digits = datasets.load_digits()

digit_test = [[0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 15, 15, 15, 0, 0, 0,
               0, 0, 0, 0, 15, 0, 0, 0,
               0, 0, 0, 0, 15, 0, 0, 0,
               0, 0, 0, 15, 0, 0, 0, 0,
               0, 0, 0, 15, 0, 0, 0, 0,
               0, 0, 0, 15, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0]]

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:], digits.target[:])

print(clf.predict(digit_test))

digit_test = np.array(digit_test)
digit_test.shape = (8, 8)
plt.imshow(digit_test, cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
