import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
def value_iteration(env, epsilon=0.0001):
    """
  :param env: the MDP
  :param epsilon: numerical precision for value function
  :return: vector representation of the value function for each state
  """
    num_s = env.num_states
    num_a = env.num_actions
    V = np.zeros(num_s)  #vector to store Value function
    while True:
        delta = 0
        for s in range(num_s):
            r = env.rewards[s]
            # print(f"reward of state s {r}")
            initial_v = V
            if s in env.terminals:
                v_best_action = r+0
            else:
                A = np.zeros(num_a)
                for a in range(num_a):
                    x=0
                    for ns in range(num_s):
                        trans_pr=env.transitions[s][a][ns]
                        # print(f"trans prob is {trans_pr} from state {s} to next state {ns} taking action {a}")
                        x+=trans_pr*initial_v[ns]
                    A[a]=x

                best_action_value=np.max(A)
                # best_action =np.argmax(A)
                # print(best_action)
                # print(A)
                v_best_action=r+ env.gamma*best_action_value
                # print(f"action matrix is {A} and best action is {best_action} and value for taking best action is {v_best_action}")

            delta = max(delta, np.abs(v_best_action - initial_v[s]))
            V[s]=v_best_action

        if delta <= epsilon*(1-env.gamma/env.gamma):
            break
    return V

def calculate_q_values(env, V=None, epsilon=0.0001):
    """
  gets q values for a markov decision process

  :param env: markov decision process
  :param epsilon: numerical precision
  :return: reurn the q values which are
  """

    #runs value iteration if not supplied as input
    if not V:
        V = value_iteration(env, epsilon)
    n = env.num_states

    Q_values = np.zeros((n, env.num_actions))
    for s in range(n):
        for a in range(env.num_actions):
            Q_values[s][a] = env.rewards[s] + env.gamma * np.dot(env.transitions[s][a], V)

    return Q_values




def get_optimal_policy(env, epsilon=0.0001, V=None):
    #runs value iteration if not supplied as input
    # if not V:
    #     V = value_iteration(env, epsilon)
    n = env.num_states
    optimal_policy = []  # our game plan where we need to

    for s in range(n):
        max_action_value = -math.inf
        best_action = 0

        for a in range(env.num_actions):
            action_value = 0.0
            for s2 in range(n):  # look at all possible next states
                action_value += env.transitions[s][a][s2] * V[s2]
                # check if a is max
            if action_value > max_action_value:
                max_action_value = action_value
                best_action = a  # direction to take
        optimal_policy.append(best_action)
    return optimal_policy


'''Policy Evaluation of the determinstic DDT policy under the GT reward'''

def policy_evaluation_GT_reward(env_gt,policy,epsilon):

    # num_a=env_gt.num_actions
    # V_t1= []
    num_s = env_gt.num_states
    V_t1 = np.zeros(num_s)
    initial_V = np.zeros(num_s)

    count=0
    while True:
        delta = 0
        # print(f"Current count {count}")
        for s in range(num_s):

            action=policy[s]
            x=0
            for ns in range(num_s):
                trasition_policy=env_gt.transitions[s][action][ns]
                x+= trasition_policy*initial_V[ns]
            V_t1[s] = env_gt.rewards[s] + env_gt.gamma * x
            # print("check delta")
            delta = max(delta, np.abs(V_t1[s] - initial_V[s]))
            initial_V[s] = V_t1[s]

        count+=1

        if delta <= epsilon*(1-env_gt.gamma/env_gt.gamma):
            break

    return V_t1


def evaluate_random_policy(env_gt,epsilon):
    num_a = env_gt.num_actions
    num_s = env_gt.num_states
    V_t1 = np.zeros(num_s)
    initial_V=np.zeros(num_s)
    action_prob = 1 / num_a
    while True:
        delta = 0
        for s in range(num_s):
            for a in range(num_a):
                x=0
                for ns in range(num_s):
                    trasition_policy = env_gt.transitions[s][a][ns]
                    x+=trasition_policy * initial_V[ns]
            V_t1[s] = env_gt.rewards[s] + env_gt.gamma *action_prob * x
            delta = max(delta, np.abs(V_t1[s] - initial_V[s]))
            initial_V[s] = V_t1[s]

        if delta <= epsilon * (1 - env_gt.gamma / env_gt.gamma):
            break
    return V_t1

class MDP:
    def __init__(self, num_rows, num_cols, terminals, rewards, gamma, noise=0.1):

        """
        Markov Decision Processes (MDP):
        num_rows: number of row in a environment
        num_cols: number of columns in environment
        terminals: terminal states (sink states)
        noise: with probability 2*noise the agent will move perpendicular to desired action split evenly,
                e.g. if taking up action, then the agent has probability noise of going right and probability noise of going left.
        """
        self.gamma = gamma
        self.num_states = num_rows * num_cols
        self.num_actions = 4  # up:0, down:1, left:2, right:3
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.terminals = terminals
        self.rewards = rewards  # think of this
        # print(f"the rewards are {self.rewards}")

        # initialize transitions given desired noise level
        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.init_transition_probabilities(noise)

    def init_transition_probabilities(self, noise):
        # 0: up, 1 : down, 2:left, 3:right

        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3
        # going UP
        for s in range(self.num_states):

            # possibility of going foward

            if s >= self.num_cols:
                self.transitions[s][UP][s - self.num_cols] = 1.0 - (2 * noise)
            else:
                self.transitions[s][UP][s] = 1.0 - (2 * noise)

            # possibility of going left

            if s % self.num_cols == 0:
                self.transitions[s][UP][s] = noise
            else:
                self.transitions[s][UP][s - 1] = noise

            # possibility of going right

            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][UP][s + 1] = noise
            else:
                self.transitions[s][UP][s] = noise

            # special case top left corner

            if s < self.num_cols and s % self.num_cols == 0.0:
                self.transitions[s][UP][s] = 1.0 - noise
            elif s < self.num_cols and s % self.num_cols == self.num_cols - 1:
                self.transitions[s][UP][s] = 1.0 - noise
            # print(self.transitions)
        # going down
        for s in range(self.num_states):

            # self.num_rows = gridHeight
            # self.num_cols = gridwidth

            # possibility of going down
            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][DOWN][s + self.num_cols] = 1.0 - (2 * noise)
            else:
                self.transitions[s][DOWN][s] = 1.0 - (2 * noise)

            # possibility of going left
            if s % self.num_cols == 0:
                self.transitions[s][DOWN][s] = noise
            else:
                self.transitions[s][DOWN][s - 1] = noise

            # possibility of going right
            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][DOWN][s + 1] = noise
            else:
                self.transitions[s][DOWN][s] = noise

            # checking bottom right corner
            if s >= (self.num_rows - 1) * self.num_cols and s % self.num_cols == 0:
                self.transitions[s][DOWN][s] = 1.0 - noise
            elif (
                    s >= (self.num_rows - 1) * self.num_cols
                    and s % self.num_cols == self.num_cols - 1
            ):
                self.transitions[s][DOWN][s] = 1.0 - noise

        # going left
        # self.num_rows = gridHeight
        # self.num_cols = gridwidth
        for s in range(self.num_states):
            # possibility of going left

            if s % self.num_cols > 0:
                self.transitions[s][LEFT][s - 1] = 1.0 - (2 * noise)
            else:
                self.transitions[s][LEFT][s] = 1.0 - (2 * noise)

            # possibility of going up

            if s >= self.num_cols:
                self.transitions[s][LEFT][s - self.num_cols] = noise
            else:
                self.transitions[s][LEFT][s] = noise

            # possiblity of going down
            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][LEFT][s + self.num_cols] = noise
            else:
                self.transitions[s][LEFT][s] = noise

            # check  top left corner
            if s < self.num_cols and s % self.num_cols == 0:
                self.transitions[s][LEFT][s] = 1.0 - noise
            elif s >= (self.num_rows - 1) * self.num_cols and s % self.num_cols == 0:
                self.transitions[s][LEFT][s] = 1 - noise

        # going right
        for s in range(self.num_states):

            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][RIGHT][s + 1] = 1.0 - (2 * noise)
            else:
                self.transitions[s][RIGHT][s] = 1.0 - (2 * noise)

            # possibility of going up

            if s >= self.num_cols:
                self.transitions[s][RIGHT][s - self.num_cols] = noise
            else:
                self.transitions[s][RIGHT][s] = noise

            # possibility of going down

            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][RIGHT][s + self.num_cols] = noise
            else:
                self.transitions[s][RIGHT][s] = noise

            # check top right corner
            if (s < self.num_cols) and (s % self.num_cols == self.num_cols - 1):
                self.transitions[s][RIGHT][s] = 1 - noise
            # check bottom rihgt corner case
            elif (
                    s >= (self.num_rows - 1) * self.num_cols
                    and s % self.num_cols == self.num_cols - 1
            ):
                self.transitions[s][RIGHT][s] = 1.0 - noise

        for s in range(self.num_states):
            if s in self.terminals:
                for a in range(self.num_actions):
                    for s2 in range(self.num_states):
                        self.transitions[s][a][s2] = 0.0
        # print(self.transitions)

    def set_rewards(self, _rewards):
        self.rewards = _rewards
        print(f"the rewards are {self.rewards}")

    def set_gamma(self, gamma):
        assert (gamma < 1.0 and gamma > 0.0)
        self.gamma = gamma


def print_array_as_grid(array_values, env):
    """
  Prints array as a grid
  :param array_values:
  :param env:
  :return:
  """
    count = 0
    for r in range(env.num_rows):
        print_row = ""
        for c in range(env.num_cols):
            print_row += "{:.2f}\t".format(array_values[count])
            count += 1
        print(print_row)


def action_to_string(act, UP=0, DOWN=1, LEFT=2, RIGHT=3):
    if act == UP:
        return "^"
    elif act == DOWN:
        return "v"
    elif act == LEFT:
        return "<"
    elif act == RIGHT:
        return ">"
    else:
        return NotImplementedError


def visualize_policy(policy, env):
    """
  prints the policy of the MDP using text arrows and uses a '.' for terminals
  """
    count = 0
    for r in range(env.num_rows):
        policy_row = ""
        for c in range(env.num_cols):
            if count in env.terminals:
                policy_row += ".\t"
            else:
                policy_row += action_to_string(policy[count]) + "\t"
            count += 1
        print(policy_row)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(1, 5, kernel_size=7).to(self.device)
        self.conv2 = nn.Conv2d(5, 1, kernel_size=5).to(self.device)
        # self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(324, 10).to(self.device)
        self.fc2 = nn.Linear(10, 1).to(self.device)

    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        # sum_rewards = 0
        # sum_abs_rewards = 0
        # x = traj.permute(0, 3, 1, 2)  # get into NCHW format
        # compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
        conv_out1 = F.leaky_relu(self.conv1(traj))
        conv_out = F.leaky_relu(self.conv2(conv_out1))
        x = conv_out.reshape((conv_out.size(dim=0),-1))
        x = F.leaky_relu(self.fc1(x))
        r = self.fc2(x)
        # sum_rewards += torch.sum(r)
        # sum_abs_rewards += torch.sum(torch.abs(r))
        return r
        # return sum_rewards, sum_abs_rewards

    def forward(self, traj_i, traj_j=None):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r = self.cum_return(traj_i)
        return cum_r