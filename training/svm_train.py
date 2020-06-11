import csv
import numpy as np
import os
import joblib
from sklearn import svm

class SVM_Train():
    def __init__(self):
        self.all_data = np.zeros([1,40])
        self.lable = [0, 1, 2, 3]
        self.all_lable = []
        self.training_data = []
        self.training_lable = []
        self.test_data = []
        self.test_lable = []
        self.dir_path = os.path.dirname(os.path.realpath(__file__))

    def load_data(self):
        for (dirpath, dirnames, filenames) in os.walk(self.dir_path):
            i = 0
            for file in filenames:
                if file.endswith('.csv'):
                    with open(self.dir_path + '/' + file) as csv_file:
                        csv_reader = csv.reader(csv_file, delimiter=',')
                        line_count = 0
                        vector = np.zeros([1, 40])
                        for row in csv_reader:
                            row = np.array(row)
                            row = row[np.newaxis, :]
                            vector = np.append(vector, row, axis=0)
                            self.all_lable.append(self.lable[i])  # Stand case with label 0
                            line_count += 1
                        print(f'Add {line_count} records as gesture samples. {self.lable[i]}')
                        vector = np.delete(vector, 0, axis=0)
                        vector = np.array(vector, dtype='float32')
                        self.all_data = np.append(self.all_data,vector,axis=0)
                    i += 1

    def trainingSVM(self):
        self.load_data()
        self.all_data = np.delete(self.all_data, 0, axis=0)
        trainingDataPorportion = 0.7
        np.random.seed(0)
        randomizedIndex = np.random.permutation(len(self.all_data))  # Generate a list of random number. All the numbers are from 0 to AllDataSize - 1
        trainingDataSize = (int)(len(self.all_data) * trainingDataPorportion)
        for i in range(trainingDataSize):
            self.training_data.append(self.all_data[randomizedIndex[i]])
            self.training_lable.append(self.all_lable[randomizedIndex[i]])

        for i in range(trainingDataSize, len(self.all_data)):
            self.test_data.append(self.all_data[randomizedIndex[i]])
            self.test_lable.append(self.all_lable[randomizedIndex[i]])
        SVM = svm.SVC(kernel='rbf', gamma=10)
        SVM.fit(self.training_data, self.training_lable)
        result = SVM.predict(self.test_data)
        joblib.dump(SVM, "train1_model.m")

# ***************************************************

if __name__ == '__main__':
    SVM_Train = SVM_Train()
    SVM = SVM_Train.trainingSVM()

