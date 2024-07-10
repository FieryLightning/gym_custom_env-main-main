import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from stable_baselines3 import A2C
import os
import v0_warehouse_robot_env # Even though we don't use this class here, we should include it here so that it registers the WarehouseRobot environment.

def run(episodes, is_training=True, render=False):

    env = gym.make('warehouse-robot-v0', render_mode='human' if render else None)

    # Divide position and velocity into segments
    # pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)    # Between -1.2 and 0.6
    # vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)    # Between -0.07 and 0.07

    # for i in env.observation_space.low:
    #     print(i)
    # for i in env.observation_space.high:
    #     print(i)


 
    pos_space_x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    pos_space_y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)
    angle_space = np.linspace(env.observation_space.low[2], env.observation_space.high[2], 8)
    target_space_x = np.linspace(env.observation_space.low[3], env.observation_space.high[3], env.unwrapped.grid_rows)
    target_space_y =  np.linspace(env.observation_space.low[4], env.observation_space.high[4], env.unwrapped.grid_cols)


    if(is_training):

        # print(len(pos_space_x))
        # print(len(pos_space_y))
        # print(len(angle_space))
        # print(len(target_space_x))
        # print(len(target_space_y))

        q = np.zeros((len(pos_space_x) + 1, len(pos_space_y) + 1, len(angle_space), len(target_space_x) + 1, len(target_space_y) + 1,  env.action_space.n)) # init a 20x20x8x4x5x4 array

        # print(len(q))
        # print(len(q[0]))
        # print(len(q[0][0]))
        # print(len(q[0][0][0]))
        # print(len(q[0][0][0][0]))

    else:
        f = open('v0_warehouse_solution.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount factor.

    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 2/episodes # epsilon decay rate
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]      # Starting position, starting velocity always 0
        # state_p = np.digitize(state[0], pos_space)
        # state_v = np.digitize(state[1], vel_space)
        
        state_x = np.digitize(state[0], pos_space_x)
        state_y = np.digitize(state[1], pos_space_y)
        state_a = np.digitize(state[2], angle_space)
        state_tx = np.digitize(state[3], target_space_x)
        state_ty = np.digitize(state[4], target_space_y)

        # print(state)

        # for x in range(50):
        #     state = env.reset()[0]      # Starting position, starting velocity always 0
        #     print(state)
        # while 1==1:
        #     continue

        terminated = False          # True when reached goal

        rewards=0

        while(not terminated and rewards > -1000):

            if is_training and rng.random() < epsilon:
                # Choose random action (0=drive left, 1=stay neutral, 2=drive right)
                action = env.action_space.sample()
            else:
                # action = np.argmax(q[state_p, state_v, :])
                action = np.argmax(q[state_x, state_y, state_a, state_tx, state_ty, :])

            new_state,reward,terminated,_,_ = env.step(action)

            # new_state_p = np.digitize(new_state[0], pos_space)
            # new_state_v = np.digitize(new_state[1], vel_space)
            
            new_state_x = np.digitize(state[0], pos_space_x)
            new_state_y = np.digitize(state[1], pos_space_y)
            new_state_a = np.digitize(state[2], angle_space)
            new_state_tx = np.digitize(state[3], target_space_x)
            new_state_ty = np.digitize(state[4], target_space_y)            

            # if is_training:
            #     q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate_a * (
            #         reward + discount_factor_g*np.max(q[new_state_p, new_state_v,:]) - q[state_p, state_v, action]
            #     )

            if is_training:
                
                # print("state_x: ", state_x)
                # print("state_y: ", state_y)
                # print("state_a: ", state_a)
                # print("state_tx: ", state_tx)
                # print("state_ty: ", state_ty)
                # print()
                # print(len(q))
                # print(len(q[0]))
                # print(len(q[0][0]))
                # print(len(q[0][0][0]))
                # print(len(q[0][0][0][0]))
                # print()
                # print()

                q[state_x, state_y, state_a, state_tx, state_ty, action] = q[state_x, state_y, state_a, state_tx, state_ty, action] + learning_rate_a * (
                    reward + discount_factor_g*np.max(q[new_state_x, new_state_y, new_state_a, new_state_tx, new_state_ty,:]) - q[state_x, state_y, state_a, state_tx, state_ty, action]
                )

            state = new_state

            # state_p = new_state_p
            # state_v = new_state_v

            state_x = new_state_x
            state_y = new_state_y
            state_a = new_state_a
            state_tx = new_state_tx
            state_ty = new_state_ty

            rewards+=reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        rewards_per_episode[i] = rewards

        print("Episode:", i + 1)

    env.close()

    # Save Q table to file
    if is_training:
        f = open('v0_warehouse_solution.pkl','wb')
        pickle.dump(q, f)
        f.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig(f'warehouse_robot.png')

if __name__ == '__main__':

    #run(1000, is_training=True, render=False)
    run(10, is_training=False, render=True)
