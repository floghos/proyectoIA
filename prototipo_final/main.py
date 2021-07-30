import os
import gym
from tf_dqn import DeepQNetwork, Agent
# from utils import plotLearning
import numpy as np
import matplotlib.pyplot as plt
import API.sg_api as api

def preprocess(observation):
    observation = observation / 255
    return np.mean(observation[30:, :], axis=2).reshape(180, 160, 1)


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
    SG_DIMS = (20, 15, STACK_SIZE)
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

    """
    print("Loading up the agent's memory with random gameplay")

    while agent.mem_cntr < 25000:
        done = False
        observation = env.reset()
        observation = preprocess(observation)
        stacked_frames = None
        observation = stack_frames(stacked_frames, observation, STACK_SIZE)
        while not done:
            action = np.random.choice([0, 1, 2])
            action += 1
            observation_, reward, done, info = env.step(action)
            observation_ = stack_frames(stacked_frames,
                                        preprocess(observation_), STACK_SIZE)
            action -= 1
            agent.store_transition(observation, action,
                                   reward, observation_, int(done))
            observation = observation_
    print("Done with random gameplay. Game on.")
    """
    EPISODES_PER_SAVE = 10
    MAX_STEPS = 101
    n_steps = 0

    for i in range(numGames):
        done = False
        #if i % 100 == 0 and i > 0:
        #    x = [j+1 for j in range(i)]
        #    plotLearning(x, scores, eps_history, filename)
        observation = api.reset()
        stacked_frames = None
        observation = stack_frames(stacked_frames, observation, STACK_SIZE)
        score = 0
        restart = False
        while not done and n_steps < MAX_STEPS:
            action = agent.choose_action(observation)
            #action += 1
            observation_, reward, done, restart = api.step(action, map)
            n_steps += 1
            observation_ = stack_frames(stacked_frames,
                                        observation_, STACK_SIZE)
            score += reward
            agent.store_transition(observation, action,
                                   reward, observation_, int(done))
            observation = observation_
            if n_steps % 4 == 0:
                agent.learn()

        if restart:
            api.restart()

        if i % EPISODES_PER_SAVE == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i-EPISODES_PER_SAVE):(i+1)])
            print('episode: ', i, 'score: ', score,
                  ' average score %.3f' % avg_score,
                  'epsilon %.3f' % agent.epsilon)
            agent.save_models()
        else:
            print('episode: ', i, 'score: ', score)
        eps_history.append(agent.epsilon)
        scores.append(score)
    # x = [i+1 for i in range(numGames)]
    # plotLearning(x, scores, eps_history, filename)
