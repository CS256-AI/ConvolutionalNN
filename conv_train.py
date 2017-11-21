import tensorflow as tf
import zener_generator as zen_gen


learning_rate = 0.001
epochs = 5
inp_feature_size = 25
threshold = 0.5
mini_batch_size = 50
regularizer = 0.001

def build_network(network_desc_file, X):
    hidden_layers = list()
    # Building the network_desc based on description file
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
                hidden_layers.append(dense_layer)
    if(len(hidden_layers)) == 0 : raise Exception("Invalid network description") # Incorrect network_desc description files
    logits = tf.layers.dense(inputs=hidden_layers[-1], units=2)
    return logits


def define_loss_function(loss_type, logits, labels):
    loss_type = loss_type.lower()
    base_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    if loss_type == "cross":
        regularization_loss = 0
    elif loss_type == "cross-l1":
        regularization_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                        if 'bias' not in v.name]) * regularizer
    elif loss_type == "cross-l2":
        regularization_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                           if 'bias' not in v.name]) * regularizer

    return base_loss + regularization_loss


def train(model_file, data_folder, optimizer, loss, accuracy, symbol_name):
    data_gen = zen_gen.DataUtil()
    data = data_gen.get_data(data_folder, symbol_name)
    # file_names, batch_x, batch_y = data

    with tf.Session() as session:
        init = tf.global_variables_initializer()
        session.run(init)
        for e in range(epochs):
            processed = 0
            print("Processing epoch {} of {}".format(e + 1, epochs))
            batched_data_list = data.get_epoch_data(mini_batch_size)
            for batch_x, batch_y in batched_data_list:
                o, l, a = session.run([optimizer, loss, accuracy], feed_dict={inp: batch_x, labels: batch_y})
                processed += mini_batch_size
                print("Processed {} training data. Batch Loss : {}. Batch Accuracy : {}".format(processed, l, a))
        print("Training complete. Final Loss: {}".format(l))

        saver = tf.train.Saver()
        if not model_file.endswith(".ckpt"): model_file += ".ckpt"
        save_path = saver.save(session, model_file)
        print("Model saved in file: ", save_path)


def test(model_file, data_folder, symbol_name):
    saver = tf.train.Saver()
    data_gen = zen_gen.DataUtil()
    data = data_gen.get_data(data_folder, symbol_name)

    with tf.Session() as session:
        # Restore variables from disk.
        if not model_file.endswith(".ckpt"): model_file += ".ckpt"
        saver.restore(session, model_file)
        print("Model restored.")
        # Check the values of the variables
        # read number of examples using the data utility
        test_x, test_y = data.get_test_data()
        a, lab, op = session.run([accuracy, labels, op_soft_max], feed_dict={inp:test_x, labels:test_y})
        print(" Labels + Prediction : ", list(zip(lab, op)))
        print("Model Accuracy : ", a)
        # print("Confusion Matrix :\n", cm)


if __name__ == '__main__':
    # input/output placeholder
    inp = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.float32)
    tf.reshape(labels, [-1, 2])

    logits = build_network("network_desc", inp)
    op_soft_max = tf.nn.softmax(logits)
    loss = define_loss_function('cross-l2', logits, labels)

    model_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    prediction = tf.equal(tf.argmax(op_soft_max, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, "float"))

    #train(model_file="D:\SJSU\Fall17\CS256\ConvolutionalNN\\model\W\model", data_folder="D:\SJSU\Fall17\CS256\cs256_hw4_data\\train_data_w", optimizer=model_optimizer, accuracy=accuracy, loss=loss, symbol_name = "W")
    test(model_file="D:\SJSU\Fall17\CS256\ConvolutionalNN\\model\W\model", data_folder="D:\SJSU\Fall17\CS256\ConvolutionalNN\\dummy", symbol_name="w")

