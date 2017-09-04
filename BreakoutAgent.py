import itertools
import numpy as np
import random
import tensorflow as tf
from collections import deque
from DSCG import DSCG
from TupleNetwork import TupleNetwork
from PIL import Image, ImageOps
import gym
import math

GAMMA = 0.99

INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.1
EXPLORE_STPES = 500000
COPY_STEPS = 2000000


INIT_REPLAY_MEMORY_SIZE = 10000
REPLAY_MEMORY_SIZE = 500000

BATCH_SIZE = 32
CODE_SIZE = 180

def random_code():
    result = np.zeros(CODE_SIZE)
    for i in range(CODE_SIZE):
        result[i] = np.random.randint(2)
    return result

def process_state(state):
    state = Image.fromarray(state)
    state = ImageOps.fit(state, (84,84), centering=(0.5,0.7))
    state = state.convert('L')      
    return np.array(state)

def get_initial_state(env):
    observation = env.reset()
    return observation

def main(_):
    env = gym.envs.make("Breakout-v0")
    dscg = DSCG(GAMMA, 4, CODE_SIZE)
    tn = TupleNetwork(CODE_SIZE, 4)

    replay_memory = deque()
    code_replay_memory = deque()
    dqn_log = deque()
    code_set = set()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    episode_reward = 0    
    episode_replay_memory = []
    epsilon = INITIAL_EPSILON

    observation = process_state(get_initial_state(env))
    state = np.stack([observation] * 4, axis=2)

    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        if random.random() <= epsilon:
            action = np.random.randint(4)
        else:    
            action = 0

        next_observation, reward, done, _ = env.step(action)
        next_observation = process_state(next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        
        episode_replay_memory.append((state, action, reward, done, next_state))
        episode_reward += reward
        
        # Current game episode is over
        if done:
            observation = process_state(get_initial_state(env))
            state = np.stack([observation] * 4, axis=2)

            for episode_replay in episode_replay_memory:
                _state, _action, _reward, _done, _next_state = episode_replay
                replay_memory.append((_state, _action, _reward, _done, _next_state, random_code()))

            dqn_log.append(episode_reward)
            if len(dqn_log) > 100:
                dqn_log.popleft()
            print ("Episode reward: ", episode_reward, " Buffer: ", len(replay_memory))
            episode_reward = 0
            episode_replay_memory = []
        else:
            state = next_state

    total_t = 0
    for episode in range(10000000):
        episode_reward = 0
        episode_replay_memory = []
        episode_code_replay_memory = []

        observation = process_state(get_initial_state(env))
        state = np.stack([observation] * 4, axis=2)
        state_code = dscg.get_codes([state])[0]

        for t in itertools.count():
            if total_t >= EXPLORE_STPES:
                code_set.add(state_code)
            if random.random() <= epsilon:
                action = np.random.randint(4)
            else:    
                if total_t < EXPLORE_STPES:
                    action = dscg.select_action(state)
                else:
                    action = tn.select_action(state_code)

            next_observation, reward, done, _ = env.step(action)
            next_observation = process_state(next_observation)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
            next_state_code = dscg.get_codes([next_state])[0]
            episode_reward += reward
                   
            if total_t < COPY_STEPS:  
                episode_replay_memory.append((state, action, reward, done, next_state))
            if total_t > EXPLORE_STPES:  
                episode_code_replay_memory.append((state_code, action, reward, done, next_state_code))

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STPES

            if (len(replay_memory) > INIT_REPLAY_MEMORY_SIZE or len(code_replay_memory) > INIT_REPLAY_MEMORY_SIZE) and total_t % 4 == 0:
                if total_t < COPY_STEPS:
                    samples = random.sample(replay_memory, BATCH_SIZE)
                else:
                    samples = random.sample(code_replay_memory, BATCH_SIZE)
                if total_t < EXPLORE_STPES:
                    state_batch = [sample[0] for sample in samples]
                    action_batch = [sample[1] for sample in samples]
                    reward_batch = [sample[2] for sample in samples]
                    done_batch = [sample[3] for sample in samples]
                    next_state_batch = [sample[4] for sample in samples]
                    random_code_batch = [sample[5] for sample in samples]
                    loss = dscg.update(sess, state_batch, action_batch, reward_batch, done_batch, next_state_batch, random_code_batch)
                elif total_t < COPY_STEPS:
                    state_batch = [sample[0] for sample in samples]
                    q_values = dscg.get_values(state_batch)
                    state_code_batch = dscg.get_codes(state_batch)

                    total_loss = 0
                    for i in range(BATCH_SIZE):
                        replay_state_code = state_code_batch[i]
                        total_loss += tn.supervise_update(replay_state_code, 0, q_values[i][0])
                        total_loss += tn.supervise_update(replay_state_code, 1, q_values[i][1])
                        total_loss += tn.supervise_update(replay_state_code, 2, q_values[i][2])
                        total_loss += tn.supervise_update(replay_state_code, 3, q_values[i][3])
                    loss = total_loss / (BATCH_SIZE * 4)
                else:
                    state_code_batch = [sample[0] for sample in samples]
                    action_batch = [sample[1] for sample in samples]
                    reward_batch = [sample[2] for sample in samples]
                    done_batch = [sample[3] for sample in samples]
                    next_state_code_batch = [sample[4] for sample in samples]

                    total_loss = 0
                    for i in range(BATCH_SIZE):
                        replay_state_code = state_code_batch[i]
                        replay_action = action_batch[i]
                        replay_reward = reward_batch[i]
                        replay_done = done_batch[i]
                        replay_next_state_code = next_state_code_batch[i]
                        total_loss += tn.update(replay_state_code, replay_action, replay_reward, replay_done, replay_next_state_code)
                    loss = total_loss / (BATCH_SIZE)
                if total_t % 1000 == 0:
                    print("loss: {}, code size: {}".format(loss, len(code_set)))
                    #print(dscg.get_codes(state_batch))
            if total_t < EXPLORE_STPES and total_t % 10000 == 0:
                dscg.update_target_network(sess)

            if done or t >= 10000:
                for episode_replay in episode_replay_memory:
                    _state, _action, _reward, _done, _next_state = episode_replay
                    replay_memory.append((_state, _action, _reward, _done, _next_state, random_code()))
                    if len(replay_memory) > REPLAY_MEMORY_SIZE:
                        replay_memory.popleft()
                for episode_replay in episode_code_replay_memory:
                    _state, _action, _reward, _done, _next_state = episode_replay
                    code_replay_memory.append((_state, _action, _reward, _done, _next_state))
                    if len(code_replay_memory) > REPLAY_MEMORY_SIZE:
                        code_replay_memory.popleft()
                if total_t > COPY_STEPS:
                    replay_memory.clear()

                dqn_log.append(episode_reward)
                if len(dqn_log) > 100:
                    dqn_log.popleft()
                print("Score: {}, Episode: {}, total_t: {}, mean: {}, std: {}".format(episode_reward, episode, total_t, np.mean(dqn_log), np.std(dqn_log)))
                with open('traning_result', 'a') as file:
                    file.writelines("{}\t{}\t{}\t{}\t{}\t{}\n".format(episode, total_t, episode_reward, len(code_set), np.mean(dqn_log), np.std(dqn_log)))
                total_t += 1
                break

            state = next_state
            state_code = next_state_code
            total_t += 1

if __name__ == '__main__':
    tf.app.run()