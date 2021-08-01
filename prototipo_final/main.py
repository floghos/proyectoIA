#import os
#import gym
import cv2
from tf_dqn import DeepQNetwork, Agent
# from utils import plotLearning
import numpy as np
import matplotlib.pyplot as plt
#from time import sleep
import API.sg_api as api

def stack_frames(stacked_frames, frame, buffer_size):
    if stacked_frames is None:
        stacked_frames = np.zeros((buffer_size, *frame.shape))
        for idx, _ in enumerate(stacked_frames):
            stacked_frames[idx, :] = frame
    else:
        stacked_frames[0:buffer_size-1, :] = stacked_frames[1:, :]
        stacked_frames[buffer_size-1, :] = frame

    stacked_frames = stacked_frames.reshape(1, *frame.shape[0:2], buffer_size)

    return stacked_frames


if __name__ == '__main__':
    #env = gym.make('Breakout-v0')
    load_checkpoint = False  # change this to True if you want to resume previous training (I think?)
    STACK_SIZE = 4
    SG_DIMS = (15, 20, STACK_SIZE)
    #original input_dims = (180, 160, 4)
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.00005, input_dims=SG_DIMS,
                  n_actions=11, mem_size=4000, batch_size=64)
    if load_checkpoint:
        agent.load_models()
    #filename = 'breakout-alpha0p000025-gamma0p9-only-one-fc-2.png'
    scores = []
    eps_history = []
    numGames = 10000 
    score = 0
    map = api.setup()

    

    EPISODES_PER_SAVE = 50
    MAX_STEPS = 150
    for i in range(numGames):
        #if i % 100 == 0 and i > 0:
        #    x = [j+1 for j in range(i)]
        #    plotLearning(x, scores, eps_history, filename)
        observation = None
        start = False
        while not start:
            observation, start = api.reset(map)
        print('Player found, starting episode...')

        stacked_frames = None
        observation = stack_frames(stacked_frames, observation, STACK_SIZE)
        
        score = 0
        n_steps = 0
        done = False
        restart = False
        # ---- Episode START ----
        while not done and n_steps < MAX_STEPS:
            #print(f'{n_steps=}')
            #print(f'{map.p_lives = }')
            action = agent.choose_action(observation)
            observation_, reward, done, restart = api.step(action, map)
            # api.render(observation_)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     done = True
            #     i = numGames+1
            #     cv2.destroyAllWindows()
            #     break
            n_steps += 1
            observation_ = stack_frames(stacked_frames,
                                        observation_, STACK_SIZE)
            score += reward

            if n_steps == MAX_STEPS:
                print('max steps reached')
                restart = True
                done = True

            agent.store_transition(observation, action,
                                   reward, observation_, int(done))
            observation = observation_
            if n_steps % 4 == 0:
                agent.learn()
        # ---- Episode END ---- 
       
        # Very important to release all pressed keys after each episode
        api.releaseAllKeys()

        if restart:
            print('Restarting...')
            api.restart()

        if i % EPISODES_PER_SAVE == 0 and i > 0:
            print('Checkpoint reached. Pausing...')
            api.pause()
            avg_score = np.mean(scores[max(0, i-EPISODES_PER_SAVE):(i+1)])
            print('episode: ', i, 'score: ', score,
                  ' average score %.3f' % avg_score,
                  'epsilon %.3f' % agent.epsilon)
            print('Saving...')
            agent.save_models()
            print('Done. Unpausing...\n')
            api.pause()
        else:
            print('episode: ', i, 'score: ', score, '\n')
        eps_history.append(agent.epsilon)
        scores.append(score)
        #sleep(1)    

    # x = [i+1 for i in range(numGames)]
    # plotLearning(x, scores, eps_history, filename)
