# export DISPLAY=:0 
import airsim
import sys
sys.path.append("../") 
import time
import numpy as np
from dqn.dqn_agent import DQNAgent
from dqn.networks import CNN, CNNTargetNetwork
from tensorboard_evaluation import *
import itertools as it
from utils import EpisodeStats
import math
from PIL import Image

last_dist = 0
reward = 0

def step_func(action):
    car_controls = interpret_action(action)
    client.setCarControls(car_controls)
    car_state = client.getCarState()
    reward = compute_reward(car_state)
    done = isDone(car_state, car_controls, reward)
    #if done == 1:
    #    reward = -10
    #    # client.reset()
    #    car_control = interpret_action(1)
    #    client.setCarControls(car_control)
    #    #print("SLEEPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")
    #    #time.sleep(2)
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
    next_state = transform_input(responses)
    return next_state, reward, done

def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, rendering=False, max_timesteps=250000, history_length=0):
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
    car_controls = interpret_action(1)
    client.setCarControls(car_controls)
    time.sleep(4)
    car_state = client.getCarState()
    print("speed00000000000000000", car_state.speed)
    
        
    car_state = client.getCarState()
    pd = car_state.kinematics_estimated.position
    car_pt = np.array([pd.x_val, pd.y_val, pd.z_val])
    print("INITIAL POS --- ", car_pt)
    tot_dist = 0
    # fix bug of corrupted states without rendering in gym environment
    #env.viewer.window.dispatch_events() 
    #print("STATE B4 PREPROCESSING",state.shape)
    #print("STATE TYPE B4 PREPROCESSING",type(state))
    # append image history to first state
    #state = state_preprocessing(state)
    
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
    print("STATE b4 PREEEEEE",len(responses))
    print("STATE b4 PREEEE TYPEEEEE",type(responses))
    state = transform_input(responses)
    print("STATE after PREPROCESSING",state.shape)
    print("STATE TYPE after PREPROCESSING",type(state))
    image_hist.extend([state] * (history_length + 1))
    print("STATE after Extend",state.shape)
    state = np.array(image_hist).reshape(84, 84, history_length + 1)
    print("STATE Final",state.shape)

    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)
        
        action_id = agent.act(state = state, deterministic=deterministic)
        #action = back_to_id(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal = step_func(action_id)
            
            reward += r

            #if rendering:
            #    env.render()

            if terminal: 
                 break

        #next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(84, 84, history_length + 1)

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
 
    tensorboard_dir_eval="./tensorboardEval"
    print("... train agent")
    tensorboard = Evaluation(tensorboard_dir, ["episode_reward","left", "right", "sharp_left", "sharp_right", "accel", "brake"])
    tensorboard_eval = Evaluation(tensorboard_dir_eval, ["eepisode_reward","eleft", "eright", "esharp_left", "esharp_right", "eaccel", "ebrake"])
    mt=250000
    #for i in range(1, num_episodes + 1):
    i=1
    while True:
        print("epsiode %d" % i)
        global last_dist
        global reward
        last_dist = 0
        reward = 0
        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
        stats = run_episode(env, agent, skip_frames=5, history_length=history_length, deterministic=False, do_training=True)
        #if i % 2 == 0 and (i > 0) and max_timesteps <= 250000:
        #    max_timesteps += 10000
            
        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                      "left" : stats.get_action_usage(5),
                                                      "right" : stats.get_action_usage(4),
                                                      "sharp_left" : stats.get_action_usage(3),
                                                      "sharp_right" : stats.get_action_usage(2),
                                                      "accel" : stats.get_action_usage(1),
                                                      "brake" : stats.get_action_usage(0)
                                                      })

        # TODO: evaluate agent with deterministic actions from time to time
        # ...
        if i % 50 == 0:
            stats = run_episode(env, agent, deterministic=True, history_length=history_length, do_training=False, rendering=True)
            tensorboard_eval.write_episode_data(i, eval_dict={ "eepisode_reward" : stats.episode_reward, 
                                                      "eleft" : stats.get_action_usage(5),
                                                      "eright" : stats.get_action_usage(4),
                                                      "esharp_left" : stats.get_action_usage(3),
                                                      "esharp_right" : stats.get_action_usage(2),
                                                      "eaccel" : stats.get_action_usage(1),
                                                      "ebrake" : stats.get_action_usage(0)
                                                      })

        if i % 50 == 0:
            agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt"))
        i+=1 

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

def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[..., :3], [0.2125, 0.7154, 0.0721])
    gray = 2 * gray.astype('float32') - 1
    return gray

def transform_input(responses):
    img1d = np.array(responses[0].image_data_float, dtype=np.float)
    img1d = 255/np.maximum(np.ones(img1d.size), img1d)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

    image = Image.fromarray(img2d)
    im_final = np.array(image.resize((84, 84)).convert('L')) 

    return im_final

def interpret_action(action):
    car_controls.brake = 0
    car_controls.throttle = 1
    if action == 0:
        car_controls.throttle = 0
        car_controls.brake = 1
    elif action == 1:
        car_controls.steering = 0
    elif action == 2:
        car_controls.steering = 0.5
    elif action == 3:
        car_controls.steering = -0.5
    elif action == 4:
        car_controls.steering = 0.25
    else:
        car_controls.steering = -0.25
    return car_controls

"""
def compute_reward(car_state):
    MAX_SPEED = 50
    MIN_SPEED = 5
    thresh_dist = 3.5
    beta = 3

    z = 0
    pts = [np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, 125, z]), np.array([0, 125, z]), np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, -128, z]), np.array([0, -128, z]), np.array([0, -1, z])]
    # print("COMPUTE REWARD -- pts ---", pts)
    pd = car_state.kinematics_estimated.position
    car_pt = np.array([pd.x_val, pd.y_val, pd.z_val])

    dist = 10000000
    for i in range(0, len(pts)-1):
        dist = min(dist, np.linalg.norm(np.cross((car_pt - pts[i]), (car_pt - pts[i+1])))/np.linalg.norm(pts[i]-pts[i+1]))
    print(" COMPUTE REWARD --- CAR PT : ", car_pt, " ------ DIST -------- ", dist)
    # print("COMPUTE REWARD -- DIST -- ", dist, " --------- CAR SPEED -------------- ", car_state.speed)
    collision_info = client.simGetCollisionInfo()
    if collision_info.has_collided:
        reward = -10
    elif dist > thresh_dist:
        reward = -3
    else:
        # reward_dist = (math.exp(-beta*dist) - 0.5)
        # reward_speed = (((car_state.speed - MIN_SPEED)/(MAX_SPEED - MIN_SPEED)) - 0.5)
        reward_dist = (math.exp(beta*dist) - 0.5)
        reward_speed = (MAX_SPEED * ((car_state.speed)/(MAX_SPEED - MIN_SPEED)) - 0.5)
        print(" reward_dist + reward_speed ------ ", reward_dist, reward_speed)
        reward = reward_dist + reward_speed

    return reward
"""

def compute_reward(car_state):
    MAX_SPEED = 300
    MIN_SPEED = 10
    thresh_dist = 3.5
    beta = 3
    global reward

    z = 0
    pts = [np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, 125, z]), np.array([0, 125, z]), np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, -128, z]), np.array([0, -128, z]), np.array([0, -1, z])]
    pd = car_state.kinematics_estimated.position
    car_pt = np.array([pd.x_val, pd.y_val, pd.z_val])

    dist = 10000000
    for i in range(0, len(pts)-1):
        dist = min(dist, np.linalg.norm(np.cross((car_pt - pts[i]), (car_pt - pts[i+1])))/np.linalg.norm(pts[i]-pts[i+1]))

    #print(dist)
    global last_dist
    advanced_dist = dist - last_dist
    print("Advanced Distance===========", advanced_dist)
    if advanced_dist == 0:
        reward -= 0.5
    else:
        reward+=advanced_dist
    if(car_state.speed >= 12.5):
        reward -= 0.25

    collision_info = client.simGetCollisionInfo()
    if collision_info.has_collided:
        reward -= 50
		
    last_dist = dist
	
    #if dist > thresh_dist:
    #    reward = -3
    #else:
    #    reward_dist = (math.exp(-beta*dist) - 0.5)
    #    reward_speed = (((car_state.speed - MIN_SPEED)/(MAX_SPEED - MIN_SPEED)) - 0.5)
    #    reward = reward_dist + reward_speed

    return reward

#def isDone(car_state, car_controls, reward):
#    done = 0
#    # print("BRAKE, SPEED, REWARD --- ", car_controls.brake, car_state.speed, reward)
#    collision_info = client.simGetCollisionInfo()
#    if collision_info.has_collided or reward <= -3:
#        done = 1
#        print("COLLISION INFOOOOOOOOO --- ", collision_info.has_collided, " -------- VS REWARDDDDDDDDDDDDDDDD ... ", reward)
#    #if car_controls.brake == 0:
#    #    if car_state.speed <= 5:
#    #        done = 1
#    return done

def isDone(car_state, car_controls, reward):
    done = 0
    if reward <= -35:
        done = 1
    collision_info = client.simGetCollisionInfo()
    if collision_info.has_collided:
        done = 1
    #if car_controls.brake == 0:
    #    if car_state.speed <= 5:
    #        done = 1
    return done

if __name__ == "__main__":

    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()
    
    SizeRows = 84
    SizeCols = 84
    NumActions = 6
    

    
    #env = gym.make('CarRacing-v0').unwrapped
    #num_actions = 5
    #state_dim = env.observation_space.shape
    #print("ACTIONS===",num_actions)
    #print("State Dim",state_dim)
    # TODO: Define Q network, target network and DQN agent
    # ...
    Q = CNN()
    Q_target = CNNTargetNetwork()
    agent = DQNAgent(Q,Q_target,NumActions)
    train_online(client, agent, num_episodes=500, history_length=4, model_dir="./models_carracing")

