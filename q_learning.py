# --- IMPORTS ---
import pandas as pd
import numpy as np
import time
import random
# from IPython.display import clear_output

# --- GLOBAL VARIABLES ---
ACTIONS     = ['up', 'down', 'left', 'right']
LENGTH      = None
N_STATES    = None
START       = None
HOLE1       = None
HOLE2       = None
HOLE3       = None
TERMINAL    = None
EPSILON     = None
MAX_EPISODE = None
GAMMA       = None
ALPHA       = None
FIRST       = True


# --- BUILD Q TABLE ---
def build_q_table():
    table = pd.DataFrame(
        np.zeros((N_STATES, len(ACTIONS))),
        columns=ACTIONS
    )
    return table


# --- EPSILON GREEDY ACTION ---
def actor(observation, q_table):
    if np.random.uniform() < EPSILON:
        state_action = q_table.loc[observation, :]
        action = np.random.choice(state_action[state_action == np.max(state_action)].index)
    else:
        action = np.random.choice(ACTIONS)
    return action


# --- VISUALIZATION ---
def update_env(state, episode, step):
    view = np.array([['_ '] * LENGTH for _ in range(LENGTH)])
    view[TERMINAL] = '* '
    view[HOLE1] = 'X '
    view[HOLE2] = 'X '
    view[HOLE3] = 'X '
    view[state] = 'o '

    out = "\n".join("".join(row) for row in view)

    # clear_output(wait=True)
    print(out)
    time.sleep(0.1)


# --- INITIALIZE ENVIRONMENT ---
def init_env():
    global HOLE1, HOLE2, HOLE3, FIRST, START, TERMINAL

    start = START
    HOLE1 = (1, 3)
    HOLE2 = (1, 1)
    HOLE3 = (2, 1)
    
    FIRST = False
    return start, False


# --- ENVIRONMENT STEP ---
def get_env_feedback(state, action):
    reward = 0.0
    end = False
    a, b = state

    if action == 'up':
        a -= 1
        if a < 0:
            a = 0
        next_state = (a, b)
    elif action == 'down':
        a += 1
        if a >= LENGTH:
            a = LENGTH - 1
        next_state = (a, b)
    elif action == 'left':
        b -= 1
        if b < 0:
            b = 0
        next_state = (a, b)
    elif action == 'right':
        b += 1
        if b >= LENGTH:
            b = LENGTH - 1
        next_state = (a, b)

    # Rewards
    if next_state == TERMINAL:
        reward = 1.0
        end = True
    elif next_state in (HOLE1, HOLE2, HOLE3):
        reward = -1.0
        end = True

    return next_state, reward, end


# --- PLAY GAME USING LEARNED POLICY ---
def playGame(q_table):
    maze_transitions = []
    state = (3, 0)
    end = False
    a, b = state
    i = 0

    while not end:
        act = actor(a * LENGTH + b, q_table)
        print("step::", i, " action ::", act)

        maze_transitions.append(act)
        next_state, reward, end = get_env_feedback(state, act)

        state = next_state
        a, b = state
        i += 1

    print("==> Game Over <==")
    return maze_transitions


# --- Q-LEARNING ---
def Qlearn():
    q_table = build_q_table()
    episode = 0

    while episode < MAX_EPISODE:
        state, end = init_env()
        step = 0
        update_env(state, episode, step)

        while not end:
            a, b = state
            s_index = a * LENGTH + b

            # Choose action
            act = actor(s_index, q_table)

            # Environment feedback
            next_state, reward, end = get_env_feedback(state, act)
            na, nb = next_state
            ns_index = na * LENGTH + nb

            # Q-learning update
            q_predict = q_table.loc[s_index, act]

            if next_state != TERMINAL:
                q_target = reward + GAMMA * q_table.iloc[ns_index].max()
            else:
                q_target = reward

            q_table.loc[s_index, act] += ALPHA * (q_target - q_predict)

            # Move to next state
            state = next_state
            step += 1
            update_env(state, episode, step)

        episode += 1

    return q_table


# --- PARAMETERS ---
LENGTH      = 4
N_STATES    = LENGTH * LENGTH
START       = (LENGTH - 1, 0)
TERMINAL    = (0, 3)
EPSILON     = .9
MAX_EPISODE = 200
GAMMA       = .9
ALPHA       = .01


# --- RUN Q-LEARNING ---
q_table = Qlearn()
print("====== Q TABLE AFTER LEARNING ======")
print(q_table)
print("\n====== ACTION TAKEN BY AGENT TO REACH THE GOAL ======")
maze_transitions = playGame(q_table)
