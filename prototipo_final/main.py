import cv2
from tf_dqn import Agent  # , DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt
import API.sg_api as api
from time import sleep
#import os
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


if __name__ == '__main__':
    load_checkpoint = False 
    STACK_SIZE = 4
    SG_DIMS = (15, 20, STACK_SIZE)
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=SG_DIMS,
                  n_actions=11, mem_size=10000, batch_size=64, 
                  q_next_dir='prototipo_final/tmp/q_next', q_eval_dir='prototipo_final/tmp/q_eval')
    if load_checkpoint:
        agent.load_models()
    scores = []
    eps_history = []
    numGames = 10000 
    score = 0
    map = api.setup()

    EPISODES_PER_SAVE = 50
    MAX_STEPS = 100
    for i in range(numGames):
        ep_stack_counter=0
        if i % 10 == 0 and i > 0:
            ep_stack_counter+=1
            x = [j+1 for j in range(i)]
            plt.figure()
            plt.plot(x, scores)     # plot rewards
            plt.plot(x,eps_history) # plot epsilon history
            plt.xlabel(' number of episodes'), plt.ylabel('scores and epsilon')
            plt.title('Learning per 100, stack '+ str(ep_stack_counter))
            plt.show()
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

        print(f'mem_counter = {agent.mem_cntr}')
        print(f'Epsilon = {agent.epsilon}')
        # ---- Episode START ----
        while not done and n_steps < MAX_STEPS:
            #print(f'{n_steps=}')
            #print(f'{map.p_lives = }')
            action = agent.choose_action(observation)
            
            observation_, reward, done, restart = api.step(action, map)

            api.render(observation_)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                done = True
                i = numGames + 1
                cv2.destroyAllWindows()
                break
            
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
        sleep(1.3)

        # plotting scores and epsilon
        x = [i+1 for i in range(numGames)]
        plt.figure()
        plt.plot(x, scores)     # plot rewards
        plt.plot(x,eps_history) # plot epsilon history
        plt.xlabel(' number of episodes'), plt.ylabel('scores and epsilon')
        plt.title('Learning graph')
        plt.show

