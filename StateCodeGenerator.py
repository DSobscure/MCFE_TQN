import tensorflow as tf

class SCG():
    def __init__(self, code_size, feature_level):
        self.code_size = code_size
        self.feature_level = feature_level
        self.state = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name='state')
        self.random_code = tf.placeholder(shape=[None, code_size], dtype=tf.float32, name='random_code')

        self.state_code = []
        self.optimize = []
        self.loss = 0
        for i in range(self.feature_level):
            state_code = self.build_network(self.state, trainable=True)      
            self.state_code.append(state_code)

            code_loss = tf.reduce_mean(tf.pow(state_code - self.random_code, 2))
            self.loss += code_loss
            self.optimize.append(tf.train.AdamOptimizer(0.0000025).minimize(code_loss))          

    def build_network(self, x, trainable=True):
        #84 84 4
        conv1_weight = tf.Variable(tf.truncated_normal([21, 21, 4, 64], stddev = 0.02), trainable = trainable)
        conv1_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)             
        conv1_hidden_sum = tf.nn.conv2d(x, conv1_weight, strides = [1,21,21,1], padding='SAME') + conv1_bias        
        conv1_hidden = tf.nn.elu(conv1_hidden_sum)
        #4 4 64
        conv2_weight = tf.Variable(tf.truncated_normal([4, 4, 64, 128], stddev = 0.02), trainable = trainable)
        conv2_bias = tf.Variable(tf.constant(0.02, shape = [128]), trainable = trainable)
        conv2_hidden_sum = tf.nn.conv2d(conv1_hidden, conv2_weight, strides = [1,4,4,1], padding='SAME') + conv2_bias
        conv2_hidden = tf.nn.elu(conv2_hidden_sum)
        #1 1 128
        conv3_weight = tf.Variable(tf.truncated_normal([1, 1, 128, self.code_size], stddev = 0.02), trainable = trainable)
        conv3_bias = tf.Variable(tf.constant(0.02, shape = [self.code_size]), trainable = trainable)       
        conv3_hidden_sum = tf.nn.conv2d(conv2_hidden, conv3_weight, strides = [1,1,1,1], padding='SAME') + conv3_bias
        conv3_hidden = tf.nn.elu(conv3_hidden_sum)
        #1 1 24
        code = tf.reshape(conv3_hidden, [-1, self.code_size])
        print("code layer shape : %s" % code.get_shape())

        return code

    def update_code(self, sess, state, random_code):
        sess.run(self.optimize, feed_dict={self.state: state, self.random_code: random_code})
        return self.loss.eval(feed_dict={self.state: state, self.random_code: random_code})

    def get_code(self, state):
        state_code = []
        for i in range(self.feature_level):
            state_code.append(self.state_code[i].eval(feed_dict={self.state: state}))

        result = []
        for i in range(len(state)):
            number = 0
            for j in range(self.feature_level):
                for k in range(self.code_size):
                    number *= 2
                    if state_code[j][i][k] > 0.5:
                        number += 1
            result.append(number)
        return result


