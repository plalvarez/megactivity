import numpy as np
import tensorflow as tf
import pandas as pd
from scipy.spatial.distance import squareform

tf.logging.set_verbosity(tf.logging.INFO)
prob_dropout = 0.3
def DropNet(features, labels, mode):
#This function defines our CNN structure.

    # Input 
    X = tf.reshape(features["x"], [-1, 102, 102, 3]) #batchsize, width, height, channels.

    # Convolutional 
    conv1 = tf.layers.conv2d(inputs=X, 
                             filters=20, 
                             kernel_size=5, 
                             padding="same", 
                             activation=tf.nn.relu) # Same size, because no stride and padding same
    dropout1 = tf.layers.dropout(inputs=conv1, rate=prob_dropout, training=True) #dropout layer
    pool1 = tf.nn.avg_pool(dropout1, [1, 3, 3, 1], [1, 3, 3, 1], "VALID" ) #avgpool layer
    #output size now [batch size, 34, 34, 20] 

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=10,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)
    #output size: [batch size, 34, 34, 10]
    dropout2 = tf.layers.dropout(inputs=conv2, rate=prob_dropout, training=True)
    pool2 = tf.nn.avg_pool(dropout2, [1, 2, 2, 1], [1, 2, 2, 1], "VALID" )
    #output size [batch size, 17, 17, 10]

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 17*17*10])
    dense1 = tf.layers.dense(inputs=pool2_flat, units=50, activation=tf.nn.relu) #this is our fully connected layer
    dropout3 = tf.layers.dropout(inputs=dense1, rate=prob_dropout, training=True) 
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout3, units=2) 

    #create dictionary with our predictions, and then return it as an EstimatorSpec with appropriate information
    predictions = {
      "classes": tf.argmax(input=logits, axis=1), #class with highest value gets picked. 
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor") #probability of each class, using softmax
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #define our loss function. Cross entropy in this case.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.Variable(0, trainable=False) #Passing global_step to minimize() will increment it at each step
        learning_rate = 0.01
        decay_step = 1
        decay_rate = 0.0001 #decay rate
        learning_rate = tf.train.inverse_time_decay(learning_rate, global_step, decay_step, decay_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.auc(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

#Now having defined the model
def main(unused_argv):
	  # Load training and eval data
	#--------------------------------------------------------
    data = pd.read_csv('meg_mci.csv')
    target = data['class'].map({2:0, 1:1}).values
    data = data.iloc[:, 2:-3]
    train_set = []
    for i in data.index:
        image = []
        for f in ['mean', 'cov', 'std']:
            cov = data.loc[i, [c for c in data if f in c]]
            cc = squareform(cov)
            cc[cc==0] = np.mean(np.mean(cc[cc!=0]))
            image.append(cc.reshape(cc.shape[0], -1, 1))
        image = np.concatenate(image, axis=2).reshape(cc.shape[0], -1, 3, 1)
        train_set.append(image)
    train_set = np.concatenate(train_set, axis=3)
    train_set = np.transpose(train_set, axes=[3, 0, 1, 2])
    train_data = train_set[:60, :, :, :]
    train_labels = np.asarray(target[:60], dtype=np.int32)
    eval_data = train_set[60:, :, :, :] # np.array
    eval_labels = np.asarray(target[60:], dtype=np.int32)
	#--------------------------------------------------------

    #Create estimator:
    print('Done')
    MCdropout = tf.estimator.Estimator(model_fn=DropNet, 
                                        model_dir="meg/")
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    #train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, #training features
                                                        y=train_labels, #training labels
                                                        batch_size=5, 
                                                        num_epochs=None, 
                                                        shuffle=True) #shuffle the data
    MCdropout.train(input_fn=train_input_fn,
                    steps=4000, #model will train for 20000 steps
                    hooks=[logging_hook]) #specify the logging hook.
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
                                                       y=eval_labels,
                                                       num_epochs=20, 
                                                       shuffle=False)
    eval_results = MCdropout.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()