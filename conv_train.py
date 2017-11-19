import tensorflow as tf
import zener_generator as zen_gen


learning_rate = 0.1
epochs = 10
inp_feature_size = 25
threshold = 0.5


def build_network(network_desc_file, X):
    # Building the network based on description file
    hidden_layers = list()
    logits = None

    with open(network_desc_file) as f:
        for line in f:
            params = line.split(" ")
            input_layer = tf.reshape(X, [-1,25,25,1]) if len(hidden_layers) == 0 else hidden_layers[-1]
            if len(params) == 2:
                # Convolutional layer description
                kernel_size, filters = int(params[0]), int(params[1])
                conv = tf.layers.conv2d(inputs=input_layer, filters=filters,
                                        kernel_size=[kernel_size, kernel_size], activation=tf.nn.relu, padding='SAME')
                max_pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2, padding='SAME')
                hidden_layers.append(conv)
                hidden_layers.append(max_pool)
            elif len(params) == 1:
                # Dense layer
                neurons = int(params[0])
                input_layer = tf.contrib.layers.flatten(input_layer)
                dense_layer = tf.layers.dense(inputs = input_layer, units=neurons, activation= tf.nn.relu)
                logits = tf.layers.dense(inputs=dense_layer, units=1)
                hidden_layers.append(dense_layer)
                break
    return logits  # Incorrect network description files


def train(model_file, data_folder, optimizer, loss, accuracy):
    data_gen = zen_gen.Data()
    data = data_gen.get_data(data_folder)

    with tf.Session() as session:
        init = tf.global_variables_initializer()
        session.run(init)
        for e in range(epochs):
            processed = 0
            print("Processing epoch {} of {}".format(e + 1, epochs))
            # batched_data_list = data.get_epoch_data(mini_batch_size)
            # for file_names, batch_x, batch_y in data:
            file_names, batch_x, batch_y = data

            o, l, a = session.run([model_optimizer, loss, accuracy], feed_dict={inp: batch_x, labels: batch_y})
            # processed += mini_batch_size
            print("Processed {} training data. Batch Loss : {}. Batch Accuracy : {}".format(processed, l, a))
        print("Training complete. Final Loss: {}".format(l))

        saver = tf.train.Saver()
        if not model_file.endswith(".ckpt"): model_file += ".ckpt"
        save_path = saver.save(session, model_file)
        print("Model saved in file: ", save_path)


if __name__ == '__main__':
    # input/output placeholder
    inp = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.float32)
    tf.reshape(labels, [-1, 1])

    logits=build_network("network_desc", inp)

    # Optimizer and Loss
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
    model_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    op = tf.nn.sigmoid(logits)
    prediction = tf.cast(tf.greater_equal(op, tf.constant(threshold, dtype=tf.float32)), "float")
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, labels), "float"))

    train(model_file="model1", data_folder="data", optimizer=model_optimizer, accuracy=accuracy, loss=loss)