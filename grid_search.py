"""
Tips for Hyperparameter Optimization

This section lists some handy tips to consider when tuning hyperparameters of your neural network.

k-fold Cross Validation. You can see that the results from the examples in this post show some variance. A default cross-validation of 3 was used, but perhaps k=5 or k=10 would be more stable. Carefully choose your cross validation configuration to ensure your results are stable.
Review the Whole Grid. Do not just focus on the best result, review the whole grid of results and look for trends to support configuration decisions.
Parallelize. Use all your cores if you can, neural networks are slow to train and we often want to try a lot of different parameters. Consider spinning up a lot of AWS instances.
Use a Sample of Your Dataset. Because networks are slow to train, try training them on a smaller sample of your training dataset, just to get an idea of general directions of parameters rather than optimal configurations.
Start with Coarse Grids. Start with coarse-grained grids and zoom into finer grained grids once you can narrow the scope.
Do not Transfer Results. Results are generally problem specific. Try to avoid favorite configurations on each new problem that you see. It is unlikely that optimal results you discover on one problem will transfer to your next project. Instead look for broader trends like number of layers or relationships between parameters.
Reproducibility is a Problem. Although we set the seed for the random number generator in NumPy, the results are not 100% reproducible. There is more to reproducibility when grid searching wrapped Keras models than is presented in this post.

"""

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD

from sklearn.model_selection import GridSearchCV, train_test_split
import numpy

numpy.random.seed(8)

def my_model(neurons):
    model = Sequential()
    model.add(Dense(6, input_dim=8, activation='relu'))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = SGD(lr=0.2, momentum=0.02)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# input data
data = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=',')
x = data[:, :8]
y = data[:, 8]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

model = KerasClassifier(build_fn=my_model, epochs=2, batch_size=10)

grid_params = {
    'neurons': list(range(3, 30, 3))
}

grid = GridSearchCV(estimator=model, param_grid=grid_params, n_jobs=1)
grid_result = grid.fit(x_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



