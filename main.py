from keras.models import Sequential
from keras.layers import Dense

import numpy
numpy.random.seed(7)

data = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')

print("Shape of data:", data.shape)

X = data[:, :8]
Y = data[:, 8]

model = Sequential()
model.add(Dense(15, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X[:700], Y[:700], batch_size=10, epochs=30)

prediction = model.predict(X[700:])
rounded = [round(x[0]) for x in prediction]

score = sum([1 if p == Y[700+i] else 0 for i, p in enumerate(rounded)])/len(Y[700:])

print(score)