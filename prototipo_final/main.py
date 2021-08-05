import cv2
from tf_dqn import Agent  # , DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt
import API.sg_api as api
from datetime import datetime
from time import sleep
import os
#import gym
# from utils import plotLearningwsk

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

abort = False

if __name__ == '__main__':
    load_checkpoint = True 
    load_past_scores = False  # do we want to continue the plot of the previous checkpoint or start a new one? 
    train = False  # do we want the agent to "train" or just exploit what it has learned?
    
    # Remember to switch the path of the save files to 'prototipo_final/tmp/*' 
    if load_past_scores and os.path.isfile('prototipo_final/learning_plots/scores_save.npy'):
        scores = np.load('prototipo_final/learning_plots/scores_save.npy')
        scores = list(scores)
        score_averages = np.load('prototipo_final/learning_plots/score_averages_save.npy')
        score_averages = list(score_averages)
        eps_history = np.load('prototipo_final/learning_plots/eps_history_save.npy')
        eps_history = list(eps_history)
        eps_hist_short = np.load('prototipo_final/learning_plots/eps_hist_short_save.npy')
        eps_hist_short = list(eps_hist_short)
        s_episode = len(eps_history)
        s_epsilon = eps_history[-1]
    else:    
        scores = []
        score_averages = []
        eps_history = []
        eps_hist_short = []
        s_epsilon = 1.0
        s_episode = 0    

    # overriding epsilon just to show of what it has learned 
    s_epsilon = 0.0  
 
    STACK_SIZE = 4
    SG_DIMS = (15, 20, STACK_SIZE)
    agent = Agent(gamma=0.99, epsilon=s_epsilon, alpha=0.00005, input_dims=SG_DIMS,
                  n_actions=11, mem_size=10000, batch_size=64, 
                  q_next_dir='prototipo_final/tmp/q_next', q_eval_dir='prototipo_final/tmp/q_eval')
    if load_checkpoint:
        agent.load_models()

    numGames = 500 
    score = 0
    print('Starting...')
    map = api.setup()

    EPISODES_PER_SAVE = 100
    MAX_STEPS = 100
    for i in range(s_episode, numGames):
        
        observation = None
        start = False
        while not start:
            observation, start = api.reset(map)
        stacked_frames = None
        observation = stack_frames(stacked_frames, observation, STACK_SIZE)
        
        score = 0
        n_steps = 0
        done = False
        restart = False

        print(f'Episode: {i}',
              f'\nEpsilon = {agent.epsilon}',
              f'\nmem_counter = {agent.mem_cntr}')
        # ---- Episode START ----
        while not done and n_steps < MAX_STEPS:
            action = agent.choose_action(observation)
            
            observation_, reward, done, restart = api.step(action, map) 

            # Following segment just renders what our NN sees 
            # api.render(observation_)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     done = True
            #     i = numGames + 1
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

            if train and (n_steps % 4 == 0):
                agent.learn()
            
            # if abort:
            #     break
                
        # ---- Episode END ---- 
       
        # Very important to release all pressed keys after each episode
        api.releaseAllKeys()

        if restart:
            print('Restarting...')
            api.restart()

        eps_history.append(agent.epsilon)
        scores.append(score)

        if i % EPISODES_PER_SAVE == 0 and i > 0:
            print('Checkpoint reached. Pausing...')
            api.pause()
            avg_score = np.mean(scores[max(0, i-EPISODES_PER_SAVE):(i+1)])
            print('score: ', score,
                  ' average score %.3f' % avg_score,
                  'epsilon %.3f' % agent.epsilon)
            print('Saving...')
            agent.save_models()

            score_averages.append(avg_score)
            eps_hist_short.append(agent.epsilon)
            np.save('prototipo_final/tmp/scores_save.npy', scores)
            np.save('prototipo_final/tmp/score_averages_save.npy', score_averages)
            np.save('prototipo_final/tmp/eps_history_save.npy', eps_history)
            np.save(
                'prototipo_final/tmp/eps_hist_short_save.npy', eps_hist_short)

            now = datetime.now()
            dt_string = now.strftime("%d-%m_%H%M%S")
            # plotting averages
            x = [(j+1)*EPISODES_PER_SAVE for j in range(int(i/EPISODES_PER_SAVE))]
            plt.figure()
            plt.plot(x, score_averages)  # plot rewards
            plt.plot(x, eps_hist_short)  # plot epsilon history (shortened)
            plt.xlabel('number of episodes'), plt.ylabel('scores and epsilon')
            plt.title(f'Scores after {i} episodes (avg per {EPISODES_PER_SAVE} games)')
            plt.savefig(
                f'prototipo_final/learning_plots/{dt_string}_first_{i}_eps_averages.png')

            # plotting everything
            # x = [j+1 for j in range(i+1)]
            # plt.figure()
            # plt.plot(x, scores)
            # plt.plot(x, eps_history)
            # plt.xlabel('number of episodes'), plt.ylabel('scores and epsilon')
            # plt.title(f'Scores after {i} episodes')
            # plt.savefig(
            #     f'prototipo_final/tmp/learning_plots/{dt_string}_first_{i}_eps_full.png')
            print('Done. Unpausing...\n')
            api.pause()
        else:
            print('score: ', score, '\n')

        # if abort:
        #     break

        sleep(1.3)

