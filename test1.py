from art.attacks import ProjectedGradientDescent, PoisoningAttackSVM, SaliencyMapMethod, SpatialTransformation
import tensorflow as tf
import numpy as np
from art.utils import load_mnist
from matplotlib import pyplot as plt
from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod, ElasticNet
from art.classifiers.scikitlearn import ScikitlearnSVC
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

model_svc = SVC()

# load mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0, x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test[0:100]
y_test = y_test[0:100]


x_train_svc = x_train.reshape(-1, 28*28)
x_test_svc = x_test.reshape(-1, 28*28)
y_train = y_train.reshape(-1, 1)
model_svc.fit(x_train_svc, y_train)


model = tf.keras.models.Sequential([
    # tf.keras.layers.Flatten(),
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=3)
#
# loss_test, accuracy_test = model.evaluate(x_test, y_test)
# print('Accuracy on test data: {:4.2f}%'.format(accuracy_test * 100))
#
# classifier = KerasClassifier(model=model, clip_values=(0, 1))
classsifierSVC = ScikitlearnSVC(model=model_svc, clip_values=(0, 1))


# attack_PGD = ProjectedGradientDescent(classifier=classifier)
# attack_PSVM = PoisoningAttackSVM(classifier=classsifierSVC, eps=.3, step=.1, x_train=x_train, y_train=y_train, x_val=x_test[101:200], y_val=None)
attack_PSVM = PoisoningAttackSVM(classifier=classsifierSVC, eps=.3, step=.1, x_train=x_train_svc, y_train=y_train, x_val=x_test[101:200], y_val=None)
#
# attack_SMM = SaliencyMapMethod(classifier=classifier)
# attack_STran = SpatialTransformation(classifier=classifier)

# test_PGD = attack_PGD.generate(x_test)
# test_PSVM = attack_PSVM.generate(x_test)
test_PSVM = attack_PSVM.generate(x_test_svc)

# test_SMM = attack_SMM.generate(x_test)
# test_STran = attack_STran.generate(x_test)



fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.imshow(test_PGD[0].reshape(28, 28))
ax2.imshow(test_PSVM[0])
ax3.imshow(test_SMM[0].reshape(28, 28))
ax4.imshow(test_STran[0].reshape(28, 28))
plt.show()