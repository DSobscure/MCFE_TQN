import tensorflow as tf
import numpy as np

class DSCG():
    def __init__(self, gamma, action_number, code_size):
        self.action_number = action_number
        self.code_size = code_size
        
        self.state = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name='state')
        self.next_state = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name='next_state')
        self.action = tf.placeholder(tf.int32, shape=[None, 1], name='action')
        self.reward = tf.placeholder(tf.float32, shape=[None, 1], name='reward')
        self.terminal_mask = tf.placeholder(tf.float32, shape=[None, 1], name='terminal_mask')
        self.random_code = tf.placeholder(shape=[None, code_size], dtype=tf.float32, name='random_code')
            
        self.beheavior_output, self.beheavior_state_code, self.beheavior_model = self.build_network(self.state, trainable=True)
        self.target_output, self.target_state_code, self.target_model = self.build_network(self.next_state, trainable=False)            
        
        action_mask = tf.one_hot(self.action, self.action_number, name='action_mask')
        action_mask = tf.reshape(action_mask, (-1, self.action_number))
        masked_q = tf.reduce_sum(action_mask * self.beheavior_output, reduction_indices=1)
        masked_q = tf.reshape(masked_q, (-1, 1))
        max_next_q = tf.reduce_max(self.target_output, reduction_indices=1)
        max_next_q = tf.reshape(max_next_q, (-1, 1))
        self.delta = self.reward + self.terminal_mask * gamma * max_next_q - masked_q

        self.q_loss = tf.reduce_mean(tf.square(self.delta), name='loss')    
        self.code_loss = tf.reduce_mean(tf.pow(self.beheavior_state_code - self.random_code, 2))  
        self.loss = self.q_loss + self.code_loss
        self.optimize = tf.train.AdamOptimizer(0.000025).minimize(self.loss)   


    def build_network(self, x, trainable=True):
        #84 84 4
        conv1_weight = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.02), trainable = trainable)
        conv1_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)
        conv1_hidden = tf.nn.relu(tf.nn.conv2d(x, conv1_weight, strides = [1,4,4,1], padding='SAME') + conv1_bias)
        #21 21 32
        conv2_weight = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.02), trainable = trainable)
        conv2_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)
        conv2_hidden = tf.nn.relu(tf.nn.conv2d(conv1_hidden, conv2_weight, strides = [1,2,2,1], padding='SAME') + conv2_bias)
        #11 11 64
        conv3_weight = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.02), trainable = trainable)
        conv3_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)
        conv3_hidden = tf.nn.relu(tf.nn.conv2d(conv2_hidden, conv3_weight, strides = [1,1,1,1], padding='SAME') + conv3_bias)
        conv3_hidden_flat = tf.reshape(conv3_hidden, [-1, 11*11*64])
        #11 11 64
        q_fc1_weight = tf.Variable(tf.truncated_normal([11*11*64, 512], stddev = 0.02), trainable = trainable)
        q_fc1_bias = tf.Variable(tf.constant(0.02, shape = [512]), trainable = trainable)        
        q_fc1_hidden = tf.nn.relu(tf.matmul(conv3_hidden_flat, q_fc1_weight) + q_fc1_bias)

        q_linear_weight = tf.Variable(tf.truncated_normal([512, self.action_number], stddev = 0.02), trainable = trainable)
        q_linear_bias = tf.Variable(tf.constant(0.02, shape = [self.action_number]), trainable = trainable)
        q_outputs = tf.matmul(q_fc1_hidden, q_linear_weight) + q_linear_bias

        conv3_hidden_patches = tf.extract_image_patches(images=conv3_hidden, ksizes=[1, 4, 4, 1], strides=[1, 4, 4, 1], rates=[1, 1, 1, 1], padding='SAME')
        conv3_hidden_patches_flat =  tf.reshape(conv3_hidden_patches, [-1, 12*12*64])
        code_splits = tf.split(conv3_hidden_patches_flat, [16*64]*9, 1)
        codes = []
        for i in range(9):
            code_fc1_weight = tf.Variable(tf.truncated_normal([4*4*64, 256], stddev = 0.02), trainable = trainable)
            code_fc1_bias = tf.Variable(tf.constant(0.02, shape = [256]), trainable = trainable)       
            code_fc1_hidden = tf.nn.relu(tf.matmul(code_splits[i], code_fc1_weight) + code_fc1_bias)

            code_fc2_weight = tf.Variable(tf.truncated_normal([256, 256], stddev = 0.02), trainable = trainable)
            code_fc2_bias = tf.Variable(tf.constant(0.02, shape = [256]), trainable = trainable)       
            code_fc2_hidden = tf.nn.relu(tf.matmul(code_fc1_hidden, code_fc2_weight) + code_fc2_bias)

            code_fc3_weight = tf.Variable(tf.truncated_normal([256, 20], stddev = 0.02), trainable = trainable)
            code_fc3_bias = tf.Variable(tf.constant(0.02, shape = [20]), trainable = trainable)       
            code_fc3_hidden = tf.nn.sigmoid(tf.matmul(code_fc2_hidden, code_fc3_weight) + code_fc3_bias)
            codes.append(code_fc3_hidden)

        code_outputs = tf.concat(codes, 1)

        print("code layer shape : %s" % code_outputs.get_shape())

        variables = {
            'conv1_weight' : conv1_weight,
            'conv1_bias' : conv1_bias,
            'conv2_weight' : conv2_weight,
            'conv2_bias' : conv2_bias,
            'conv3_weight' : conv3_weight,
            'conv3_bias' : conv3_bias,
            'q_fc1_weight' : q_fc1_weight,
            'q_fc1_bias' : q_fc1_bias,
            'q_linear_weight' : q_linear_weight,
            'q_linear_bias' : q_linear_bias
        }
        return q_outputs, code_outputs, variables

    def get_values(self, state):
        return self.beheavior_output.eval(feed_dict={self.state : state})

    def select_action(self, state):
        values = self.get_values([state])[0]
        return np.argmax(values);

    def get_codes(self, state):
        state_code = self.beheavior_state_code.eval(feed_dict={self.state: state})

        result = []
        for i in range(len(state)):
            number = 0
            for j in range(self.code_size):
                number *= 2
                if state_code[i][j] > 0.5:
                    number += 1
            result.append(number)
        return result

    def update(self, sess, state, action, reward, terminal, next_state, random_code):
        action = np.reshape(action, (-1, 1))
        reward = np.reshape(reward, (-1, 1))
        terminal = np.reshape(terminal, (-1, 1))

        terminal_mask = np.invert(terminal) * 1

        loss, _ = sess.run([self.loss, self.optimize], feed_dict={
            self.state : state,
            self.next_state : next_state,
            self.action : action,
            self.reward : reward,
            self.terminal_mask : terminal_mask,
            self.random_code : random_code})
        return loss

    def update_target_network(self, sess):
        updates = []
        for key, value in self.beheavior_model.items():
            updates.append(self.target_model[key].assign(value))
        sess.run(updates);