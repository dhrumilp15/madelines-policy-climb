from image_client import ImageClient
import os
import torch
from torch import nn
from collections import deque, namedtuple
import random
import math
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

    def push(self, prev_state, prev_move, frame, prev_reward):
        """Save a transition"""
        prev_state = torch.frombuffer(prev_state, dtype=torch.uint8).float().reshape((1, 3, 640, 480)).to(self.device)
        prev_move = torch.tensor([[prev_move]], device=self.device)
        if frame is not None:
            frame = torch.frombuffer(frame, dtype=torch.uint8).float().reshape((1, 3, 640, 480)).to(self.device)
        prev_reward = torch.tensor(prev_reward, device=self.device)
        self.memory.append(Transition(prev_state, prev_move, frame, prev_reward))
        

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1089536, 512), # 1089536 is the output size of the last conv layer
            nn.ReLU(),
            nn.Linear(512, n_actions)
        ).to(self.device)
    
    def forward(self, x):
        return self.model(x)
    

class DQNModel:

    def __init__(self, checkpoint_file = None, image_size = 640 * 480 * 3):
        self.checkpoint_file = checkpoint_file
        self.output_folder = "dqn_output/"
        os.makedirs(self.output_folder, exist_ok=True)
        self.action_space = 2 ** 7  # number of possible actions
        
        if checkpoint_file and os.path.exists(checkpoint_file):
            print("Loading checkpoint from:", checkpoint_file)
            self.policy_net = torch.load(checkpoint_file + "_policy_net.pth")
            self.target_net = torch.load(checkpoint_file + "_target_net.pth")
        else:
            print("No checkpoint found, initializing new model.")
            self.policy_net = DQN(self.action_space)
            self.target_net = DQN(self.action_space)
        
        self.image_size = image_size
        self.client = ImageClient()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self.num_episodes = 1000

        self.memory = ReplayMemory(10000)
        self.batch_size = 32
        self.gamma = 0.99  # discount factor
        self.criterion = nn.MSELoss()
        self.steps_done = 0
        
        self.target_update = 10
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.epsilon = self.EPS_START
        self.TAU = 0.005
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action = self.policy_net(state_batch)
        state_action_values = state_action.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def select_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.frombuffer(state, dtype=torch.uint8).float().reshape((1, 3, 640, 480)).to(self.device)
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        eps_threshold = 0
        if sample <= eps_threshold:
            # random action
            return torch.randint(0, self.action_space, device=self.device, dtype=torch.long).view(1, 1)
        else:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)

    def update_model_params(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)
    
    def run(self):
        for i_episode in range(self.num_episodes):
            # Initialize the environment and get its state
            self.client.process_frame_and_send_move(self.select_action, self.optimize_model, self.update_model_params, self.memory)


if __name__ == "__main__":
    # while True:
    #     try:
    #         model = DQNModel()
    #         model.run()
    #     except KeyboardInterrupt:
    #         break
    #     except Exception as e:
    #         print(e)
    model = DQNModel()
    model.run()