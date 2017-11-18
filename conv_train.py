import tensorflow as tf


op_classes = 10
learning_rate = 0.01

X = tf.placeholder()
Y = tf.placeholder()

labels = tf.placeholder()

hidden_layers = list()
logits = None


def build_network(network_desc_file):
    # Building the network based on description file
    with open(network_desc_file) as f:
        global X
        for line in f:
            params = line.split(" ")
            input_layer = X if len(hidden_layers) == 0 else input_layer = hidden_layers[-1]
            if len(params) == 2:
                # Convolutional layer description
                kernel_size, filters = int(params[0]), int(params[1])
                conv = tf.layers.conv2d(inputs=input_layer, filters= filters,
                                        kernel_size= [kernel_size, kernel_size], activation= tf.nn.relu)
                max_pool = tf.layers.max_pooling2d(inputs = conv, pool_size=[2,2], strides=2)
                hidden_layers.append(conv)
                hidden_layers.append(max_pool)
            elif len(params) == 1:
                # Dense layer
                neurons = int(params[0])
                input_layer = tf.contrib.layers.flatten(input_layer)
                dense_layer = tf.layers.dense(inputs = input_layer, units=neurons, activation= tf.nn.relu)
                logits = tf.layers.dense(inputs = dense_layer, units=1)
                hidden_layers.append(dense_layer)
                break


# Defining loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits))
tf.nn.si

# Defining optimizer and training
training = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

