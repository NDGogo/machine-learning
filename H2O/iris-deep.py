import h2o

# Use all available computing resources
h2o.init()

# Import data to H2O cluster
datasets = "https://raw.githubusercontent.com/DarrenCook/h2o/bk/datasets/iris_wheader.csv"
data = h2o.import_file(datasets)

# Remove label column and leave only features
y = "class"
x = data.names
x.remove(y)

# Create training and testing sets 
train, test = data.split_frame([0.8])

#  Training the model
m = h2o.estimators.deeplearning.H2ODeepLearningEstimator()
m.train(x, y, train)

# Using the model
p = m.predict(test)

print(m.confusion_matrix(train))
print(p)