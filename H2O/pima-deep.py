# Replication of Reena's article: https://www.kdnuggets.com/2018/01/deep-learning-h2o-using-r.html

import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

h2o.init ()
h2o.remove_all()

# Preparation of data
dataset = 'D:/Repositories/machine-learning/Datasets/pima-indians-diabetes.csv'
pima = h2o.import_file(path=dataset)

label = "C9"
features = pima.names
features.remove(label)

train, valid = pima.split_frame(ratios=[0.75], seed=33)
train[label] = train[label].asfactor()
valid[label] = valid[label].asfactor()

#  Training the model
model = H2ODeepLearningEstimator(activation="rectifier_with_dropout", hidden=[190,63,21,7], epochs=50, input_dropout_ratio=0.1)
model.train(x=features, y=label, training_frame=train, validation_frame=valid)

# Validating the model
predictions = model.predict(valid)
print(predictions)
print(model.confusion_matrix(valid=True))