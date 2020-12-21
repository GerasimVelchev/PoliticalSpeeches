# -*- coding: utf-8 -*-

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from decimal import Decimal
from sklearn.svm import SVC
import codecs, sys
import csv

sys.stdout = codecs.getwriter("cp1251")(sys.stdout, 'xmlcharrefreplace')
log_file_name = "classification_calculation.log"
log = codecs.open(log_file_name, 'w', encoding = "utf-8")

features_file_name = "stylometric_features.csv"
features_file = open(features_file_name, encoding = "utf-8")
csv_reader = csv.reader(features_file)

feature_names = ['mean_word_length', 'std_word_length', 'lexical_diversity', 'noun_count', 'adjective_count',
                 'verb_count', 'pos_tag_entropy', 'person_entities_count', 'speaker']

X = []
y = []

for row in csv_reader:
    to_skip = 0
    for token in row:
        if token in feature_names:
            to_skip = 1
    if to_skip == 1:
        continue

    attributes_without_last = row[2:-1]
    attibutes_without_last = list(map(lambda x: Decimal(x), attributes_without_last))
    class_attribute = row[8]

    X.append(attributes_without_last)
    y.append(class_attribute)

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

log.write(str(len(X_train)) + "\n" + str(len(X_test)) + "\n")

svm_model_ovr_rbf = OneVsRestClassifier(SVC(kernel = 'rbf', C = 1)).fit(X_train, y_train)
svm_model_ovr_rbf_predictions = svm_model_ovr_rbf.predict(X_test)
# model accuracy for X_test
accuracy = svm_model_ovr_rbf.score(X_test, y_test)

# creating a confusion matrix
# conf_matrix = confusion_matrix(y_test, svm_predictions)

log.write("Accuracy svm_model_ovr_rbf: " + str(accuracy) + "\n")
# log.write(str(conf_matrix))

svm_model_ovo_linear = OneVsOneClassifier(SVC(kernel = 'linear', C = 1)).fit(X_train, y_train)
svm_model_linear_predictions = svm_model_ovo_linear.predict(X_test)
accuracy = svm_model_ovo_linear.score(X_test, y_test)
log.write("Accuracy svm_model_ovo_linear: " + str(accuracy) + "\n")

# creating a confusion matrix
# conf_matrix = confusion_matrix(y_test, svm_predictions)

# log_reg_model_ovr = OneVsRestClassifier(linear_model.SGDClassifier(max_iter = 1000, tol = 1e-3)).fit(X_train, y_train)
# accuracy = log_reg_model_ovr.score(X_test, y_test)
# log.write("Accuracy log_reg_model_ovr: " + str(accuracy) + "\n")
