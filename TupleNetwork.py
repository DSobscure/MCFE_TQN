from TupleFeature import TupleFeature
import numpy as np

LEARNING_RATE = 1.0
GAMMA = 0.99
FEATURE_BLOCK_COUNT = 9

class TupleNetwork(object):
    def __init__(self, code_size, action_count):
        self.code_size = code_size
        self.action_count = action_count

        self.q_featureSets = []
        for i in range(action_count):
            q_featureSet = []
            for i in range(FEATURE_BLOCK_COUNT):
                q_featureSet.append(TupleFeature(20, i))
            self.q_featureSets.append(q_featureSet)
    
    def get_q_value(self, state, action):
        sum = 0
        for i in range(FEATURE_BLOCK_COUNT):
            sum += self.q_featureSets[action][i].get_score(state)
        return sum

    def update_q(self, state, action, delta):
        for i in range(FEATURE_BLOCK_COUNT):
            self.q_featureSets[action][i].update_score(state, delta / FEATURE_BLOCK_COUNT)

    def select_action(self, state):
        return np.argmax([self.get_q_value(state, action) for action in range(self.action_count)])

    def update(self, state, action, reward, done, next_state):
        if done:
            q_delta = LEARNING_RATE * (reward - self.get_q_value(state, action))
        else:
            next_max_q = GAMMA * np.max([self.get_q_value(next_state, action) for action in range(self.action_count)])
            q_delta = LEARNING_RATE * (reward + next_max_q - self.get_q_value(state, action))
        self.update_q(state, action, q_delta)
        return q_delta ** 2

    def supervise_update(self, state, action, value):
        q_delta = LEARNING_RATE * (value - self.get_q_value(state, action))
        self.update_q(state, action, q_delta)
        return q_delta ** 2
            

