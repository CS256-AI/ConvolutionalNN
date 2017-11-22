import tensorflow as tf
import zener_generator as zen_gen
import matplotlib.pyplot as p
import time

learning_rate = 0.001
num_epochs = 5
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
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=regularizer, scope=None)
        weights = [v for v in tf.trainable_variables() if 'bias' not in v.name]  # all vars of your graph
        regularization_loss = tf.contrib.layers.apply_regularization(l1_regularizer, weights)

    elif loss_type == "cross-l2":
        regularization_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                           if 'bias' not in v.name]) * regularizer

    return base_loss + regularization_loss


def train(model_file, data_folder, optimizer, loss, accuracy, symbol_name, epochs):
    start_time = time.time()
    data_gen = zen_gen.DataUtil()
    data = data_gen.get_data(data_folder, symbol_name)
    data_size = len(data.total_data)
    train_x, train_y = data.get_test_data()
    snapshot = list()

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
                snapshot.append(( e*data_size+processed, l))
        t_l, t_a = session.run([loss, accuracy], feed_dict={inp: train_x, labels: train_y})
        print("Training complete. Final Training Loss: {}, Final Training accuracy: {}".format(t_l, t_a))

        saver = tf.train.Saver()
        if not model_file.endswith(".ckpt"): model_file += ".ckpt"
        save_path = saver.save(session, model_file)
        print("Model saved in file: ", save_path)
    end_time = time.time()
    print("Time for training : {} s".format(end_time - start_time))
    return snapshot

def five_fold_train(model_file, data_folder, optimizer, loss, accuracy, symbol_name, epochs):
    start_time = time.time()
    data_gen = zen_gen.DataUtil()
    total_data = data_gen.get_data(data_folder, symbol_name).total_data
    train_loss = 0.0; valid_loss = 0.0; train_acc = 0.0; valid_acc = 0.0
    with tf.Session() as session:
        init = tf.global_variables_initializer()
        session.run(init)
        fold = len(total_data)/5
        valid_fold_start, valid_fold_end = 0, fold
        for f in range(5):
            print("Training data by leaving out fold {}".format(f+1))
            train_data = zen_gen.Data(total_data[:valid_fold_start] + total_data[valid_fold_end:])
            valid_data = zen_gen.Data(total_data[valid_fold_start:valid_fold_end])
            train_x, train_y = train_data.get_test_data() #Complete training data
            valid_x, valid_y = valid_data.get_test_data()
            # read number of examples using the data utility
            for e in range(epochs):
                processed = 0
                print("Processing epoch {} of {}".format(e+1, epochs))
                # batch_iters = len(train_data)/mini_batch_size
                for batch_x, batch_y in train_data.get_epoch_data(mini_batch_size):
                    b_o, b_l, b_a = session.run([optimizer, loss, accuracy], feed_dict={inp: batch_x, labels: batch_y})
                    processed += mini_batch_size
                    print("Processed {} training data. Current Loss : {}. Batch Accuracy : {}".format(processed, b_l, b_a))
            # Get the training loss and accuracy
            t_l, t_a = session.run([loss, accuracy], feed_dict={inp: train_x, labels: train_y})
            print("Training complete by leaving out fold {}. Loss: {}. Accuracy: {}".format(f + 1, t_l, t_a))
            # Get the validation loss and accuracy
            v_l, v_a = session.run([loss, accuracy], feed_dict={inp: valid_x, labels: valid_y})
            print("Training complete by leaving out fold {}. Loss: {}. Accuracy: {}".format(f + 1, v_l, v_a))
            train_loss += t_l; valid_loss += v_l
            train_acc += t_a; valid_acc += v_a
            valid_fold_start += fold
            valid_fold_end += fold
        train_loss = train_loss/5; valid_loss = valid_loss/5
        train_acc = train_acc / 5; valid_acc = valid_acc / 5
        print(' Average training loss : {}, Average validation loss : {}'.format(train_loss, valid_loss))
        print(' Average training accuracy : {}, Average validation accuracy : {}'.format(train_acc, valid_acc))
        saver = tf.train.Saver()
        if not model_file.endswith(".ckpt"): model_file+= ".ckpt"
        save_path = saver.save(session, model_file)
        print("Model saved in file: ", save_path)
        end_time = time.time()
        print("Time for five-fold training : {} s".format(end_time - start_time))
        return (train_loss,valid_loss)

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
        # print(" Labels + Prediction : ", list(zip(lab, op)))
        print("Model Accuracy : ", a)
        return a
        # print("Confusion Matrix :\n", cm)


def experiment_1(model_dest, data_folder, test_folder, optimizer, accuracy, loss, symbol):
    """
    Experiment compares the training and validation loss of the model for various max updates(epochs) count

    """
    training_cost, validation_cost, test_accuracy, max_updates = list(), list(), list(), list()
    for epochs in range(2,31):
        max_updates.append(epochs)
        model_file = model_dest+"_"+str(epochs)+"_"+symbol
        t_cost, v_cost = five_fold_train(model_file=model_file, data_folder=data_folder,
                               optimizer=optimizer, loss=loss, accuracy= accuracy, symbol_name=symbol, epochs=epochs)
        t_accuracy = test(model_file=model_file, data_folder=data_folder, symbol_name=symbol)
        training_cost.append(t_cost)
        validation_cost.append(v_cost)
        test_accuracy.append(t_accuracy)

    # Plotting results
    p.subplot(2, 1, 1)
    p.title("Training & Validation Cost vs Max-updates")
    p.xlabel("Max-updates")
    p.ylabel("Cost")
    p.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
    p.plot(max_updates, training_cost, 'b-', label="Training cost")
    p.plot(max_updates, validation_cost, 'g-', label="Validation cost")
    p.legend()
    p.subplot(2, 1, 2)
    p.title("Test Accuracy of Models")
    p.xlabel("Model No")
    p.ylabel("Accuracy")
    p.plot(max_updates, test_accuracy, 'r-', label="Test Accuracy")
    p.tight_layout()
    p.show()


def experiment_2(model_dest, data_folder, optimizer, accuracy, symbol, num_epochs):
    loss = define_loss_function('cross', logits, labels)
    loss_l1 = define_loss_function('cross-l1', logits, labels)
    loss_l2 = define_loss_function('cross-l2', logits, labels)

    model_dest = model_dest + "/{}_{}_{}/"

    result_l1 = train(model_file=model_dest.format(symbol, "l1", num_epochs),
                      data_folder=data_folder, optimizer=model_optimizer,
                      accuracy=accuracy, loss=loss_l1, symbol_name=symbol, epochs=num_epochs)

    result_l2 = train(model_file=model_dest.format(symbol, "l2", num_epochs),
                      data_folder=data_folder, optimizer=model_optimizer,
                      accuracy=accuracy, loss=loss_l2, symbol_name=symbol, epochs=num_epochs)

    result = train(model_file=model_dest.format(symbol, "ll", num_epochs),
                   data_folder=data_folder, optimizer=model_optimizer,
                   accuracy=accuracy, loss=loss, symbol_name=symbol, epochs=num_epochs)

    x, y = zip(*result_l1)
    x, y2 = zip(*result_l2)
    x, y3 = zip(*result)

    p.title("Convergence")
    p.xlabel("Data")
    p.ylabel("Cost")
    p.plot(x, y, 'b-', label="L1 regularization")
    p.plot(x, y2, "g-", label="L2 regularization")
    p.plot(x, y3, "r-", label="No regularization")
    p.legend()
    p.show()


if __name__ == '__main__':
    # input/output placeholder
    inp = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.float32)
    tf.reshape(labels, [-1, 2])

    logits = build_network("network_desc", inp)
    op_soft_max = tf.nn.softmax(logits)

    loss = define_loss_function('cross', logits, labels)
    model_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    prediction = tf.equal(tf.argmax(op_soft_max, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, "float"))


    experiment_2("D:\SJSU\Fall17\CS256\ConvolutionalNN\model",
                 "D:\SJSU\Fall17\CS256\cs256_hw4_data\\train_data_w", optimizer=model_optimizer,
                 accuracy=accuracy, symbol="W", num_epochs=num_epochs)

    #five_fold_train(model_file="model_s_fivefold_exp2_1l", data_folder="/Users/rahuldalal/train_data_1k", optimizer=model_optimizer, accuracy=accuracy, loss=loss, symbol_name="s", epochs=10)
    #test(model_file="model_s_train_exp2_1l", data_folder="/Users/rahuldalal/test_data", symbol_name="s")
    #test(model_file="model_s_fivefold_exp2_1l", data_folder="/Users/rahuldalal/test_data", symbol_name="s")
