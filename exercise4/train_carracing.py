# export DISPLAY=:0 

import sys
sys.path.append("../") 
import time
import numpy as np
import gym
from dqn.dqn_agent import DQNAgent
from dqn.networks import CNN, CNNTargetNetwork
from tensorboard_evaluation import *
import itertools as it
from utils import EpisodeStats

LEFT = 1
RIGHT = 2
STRAIGHT = 0
ACCELERATE = 3
BRAKE = 4

def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, rendering=False, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()
   
    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events() 

    # append image history to first state
    state = state_preprocessing(state)
    
    image_hist.extend([state] * (history_length + 1))
    
    state = np.array(image_hist).reshape(96, 96, history_length + 1)
    
    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)
        
        action_id = agent.act(state = state, deterministic=deterministic)
        action = back_to_id(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            
            reward += r

            if rendering:
                env.render()

            if terminal: 
                 break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if terminal or (step * (skip_frames + 1)) > max_timesteps : 
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, history_length=0, model_dir="./models_carracing", tensorboard_dir="./tensorboard"):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")
    tensorboard = Evaluation(tensorboard_dir, ["episode_reward", "straight", "left", "right", "accel", "brake"])
    max_timesteps = 50
    for i in range(num_episodes):
        print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
        
        stats = run_episode(env, agent, max_timesteps=max_timesteps,skip_frames=5, history_length=history_length, deterministic=False, do_training=True)
        if i % 20 == 0 and (i > 0) and max_timesteps <= 500:
            max_timesteps += 50
            
        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                      "straight" : stats.get_action_usage(STRAIGHT),
                                                      "left" : stats.get_action_usage(LEFT),
                                                      "right" : stats.get_action_usage(RIGHT),
                                                      "accel" : stats.get_action_usage(ACCELERATE),
                                                      "brake" : stats.get_action_usage(BRAKE)
                                                      })

        # TODO: evaluate agent with deterministic actions from time to time
        # ...
        if i % 10 == 0 or (i >= num_episodes - 1):
            stats = run_episode(env, agent, deterministic=True, max_timesteps=max_timesteps, history_length=history_length, do_training=False, rendering=True)
        
        
        if i % 100 == 0 or (i >= num_episodes - 1):
            agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt")) 

    tensorboard.close_session()

def back_to_id(a):
    if (a == LEFT):
        return [-1.0, 0.0, 0.0]  # LEFT: 1
    elif (a == RIGHT):
        return [1.0, 0.0, 0.0]  # RIGHT: 2
    elif (a == ACCELERATE):
        return [0.0, 1.0, 0.0]  # ACCELERATE: 3
    elif (a == BRAKE):
        return [0.0, 0.0, 0.2]  # BRAKE: 4
    else:
        return [0.0, 0.0, 0.0]  # STRAIGHT = 0 ACCELERATE ALL TIME
    
    
def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[..., :3], [0.2125, 0.7154, 0.0721])
    gray = 2 * gray.astype('float32') - 1
    return gray

if __name__ == "__main__":

    env = gym.make('CarRacing-v0').unwrapped
    num_actions = 5
    state_dim = env.observation_space.shape
    print("ACTIONS===",num_actions)
    print("State Dim",state_dim)
    # TODO: Define Q network, target network and DQN agent
    # ...
    Q = CNN()
    Q_target = CNNTargetNetwork()
    agent = DQNAgent(Q,Q_target,num_actions)
    # train_online(env, agent, num_episodes=1000, history_length=4, model_dir="./models_carracing")
    train_online(env, agent, num_episodes=900, history_length=4, model_dir="./models_carracing")

