"""
@author Wildo Monges
Note:
    I've used the code from https://www.tensorflow.org/get_started/eager
    to learn and resolve iris classification problem using tensorflow
    for practice purpose
"""

from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

print("Tensorflow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Iris classification problem
train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))


def parse_csv(line):
    example_defaults = [[0.], [0.], [0.], [0.], [0]]
    parsed_line = tf.decode_csv(line, example_defaults)
    features = tf.reshape(parsed_line[:-1], shape=(4,))
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label


train_dataset = tf.data.TextLineDataset(train_dataset_fp)
# Skip the first row (header)
train_dataset = train_dataset.skip(1)
# Parse each row
train_dataset = train_dataset.map(parse_csv)
# Randomize
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.batch(32)

# View a single example entry from a batch
features, label = iter(train_dataset).next()
print("Example features:", features[0])
print("Example label:", label[0])

model = tf.keras.Sequential(
    [
        # 4 is the amount of features [SepalLength, SepalWidth, PetalLength, PetalWidth]
        tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(3)
    ]
)


def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# Keep results for plotting
train_loss_results = []
train_accuracy_results = []
# An epoch is one pass through the dataset
num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()
    # Training loop - using batches of 32
    for x, y in train_dataset:
        # Optimize the model
        grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())
        # Track progress
        # Add current batch loss
        epoch_loss_avg(loss(model, x, y))
        # Compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle("Training Metrics")
axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()


test_url = "http://download.tensorflow.org/data/iris_test.csv"
test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url), origin=test_url)
test_dataset = tf.data.TextLineDataset(test_fp)
test_dataset = test_dataset.skip(1)
test_dataset = test_dataset.map(parse_csv)
test_dataset = test_dataset.shuffle(1000)
test_dataset = test_dataset.batch(32)


test_accuracy = tfe.metrics.Accuracy()
for (x, y) in test_dataset:
    prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


# Test with some data
class_ids = ["Iris setosa", "Iris versicolor", "Iris virginica"]
predict_dataset = tf.convert_to_tensor(
    [
        [5.1, 3.3, 1.7, 0.5, ],
        [5.9, 3.0, 4.2, 1.5, ],
        [6.9, 3.1, 5.4, 2.1]
    ]
)
predictions = model(predict_dataset)
for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    name = class_ids[class_idx]
    print("Example {} prediction: {}".format(i, name))
