import warnings
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from torch.distributions import Normal

from utils import ReplayBuffer, get_env, run_episode

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

GAUSSIAN_POLICY = True
HIDDEN_SIZE = 256
HIDDEN_LAYERS = 1
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
ALPHA_LR = 1e-3
GAMMA = 0.99
TAU = 0.005
LOG_STD_MIN = -5
LOG_STD_MAX = 2


class NeuralNetwork(nn.Module):
    """
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int,
                 hidden_layers: int, activation: str):
        super(NeuralNetwork, self).__init__()

        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU() if activation == 'relu' else nn.Tanh(),
            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU() if activation == 'relu' else nn.Tanh())
              for _ in range(hidden_layers)]
        )

        self.output_layer = nn.Linear(hidden_size, output_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layers(s)
        return self.output_layer(x)


class Actor:
    def __init__(self, hidden_size: int, hidden_layers: int, actor_lr: float,
                 state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.optimizer = None
        self.policy = None
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = LOG_STD_MIN
        self.LOG_STD_MAX = LOG_STD_MAX
        self.setup_actor()

    def setup_actor(self):
        """
        This function sets up the actor network in the Actor class.
        """
        self.policy = NeuralNetwork(self.state_dim, 2 * self.action_dim, self.hidden_size, self.hidden_layers,
                                    'relu').to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.actor_lr)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        """
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        """
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor,
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        """
        @param state: torch.Tensor, state of the agent
        @param deterministic:  boolean, if true return a deterministic action  otherwise sample from the policy
         distribution.
        @return: action sampled from the policy distribution and log probability of the action
        """

        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'

        # Get mean and log_std from the policy network. Split the output into mean and log_std
        output = self.policy(state)
        if len(output.shape) == 1:
            mean, log_std = torch.split(output, self.action_dim, dim=0)
        else:
            # We have a batch of states
            mean, log_std = torch.split(output, self.action_dim, dim=1)

        # Clamp log_std
        log_std = self.clamp_log_std(log_std)
        std = torch.exp(log_std)
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(mean.shape)
        else:
            normal = Normal(mean, std)
            sample = normal.rsample()
            action = torch.tanh(sample)
            log_prob = normal.log_prob(sample) - torch.log(1 - action.pow(2) + 1e-7)

        return action, log_prob


class Critic:
    def __init__(self, hidden_size: int,
                 hidden_layers: int, critic_lr: float, state_dim: int = 3,
                 action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.Q = None
        self.optimizer = None
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        self.Q = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers,
                               'relu').to(self.device)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.critic_lr)


class TrainableParameter:
    """
    This class could be used to define a trainable parameter in your method. You could find it
    useful if you try to implement the entropy temperature parameter for SAC algorithm.
    """

    def __init__(self, init_param: float, lr_param: float,
                 train_param: bool, device: torch.device = torch.device('cpu')):
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.actor = None
        self.critic1 = None
        self.critic2 = None
        self.target_critic1 = None
        self.target_critic2 = None
        self.alpha = None
        self.target_entropy = None
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)

        self.setup_agent()

    def setup_agent(self):
        # Hint: You may find the TrainableParameter class useful for implementing the entropy temperature parameter.
        self.actor = Actor(HIDDEN_SIZE, HIDDEN_LAYERS, ACTOR_LR, self.state_dim, self.action_dim, self.device)
        self.critic1 = Critic(HIDDEN_SIZE, HIDDEN_LAYERS, CRITIC_LR, self.state_dim, self.action_dim, self.device)
        self.critic2 = Critic(HIDDEN_SIZE, HIDDEN_LAYERS, CRITIC_LR, self.state_dim, self.action_dim, self.device)
        self.target_critic1 = Critic(HIDDEN_SIZE, HIDDEN_LAYERS, CRITIC_LR, self.state_dim, self.action_dim,
                                     self.device)
        self.target_critic2 = Critic(HIDDEN_SIZE, HIDDEN_LAYERS, CRITIC_LR, self.state_dim, self.action_dim,
                                     self.device)
        self.alpha = TrainableParameter(1, ALPHA_LR, True, self.device)
        # Copy the critic parameters to target critic
        self.critic_target_update(self.critic1.Q, self.target_critic1.Q, 1.0, False)
        self.critic_target_update(self.critic2.Q, self.target_critic2.Q, 1.0, False)
        # Target entropy
        self.target_entropy = -self.action_dim

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray, action to apply on the environment, shape (1, _)
        """
        # Get action from the actor network
        with torch.no_grad():
            action, _ = self.actor.get_action_and_log_prob(torch.tensor(s, dtype=torch.float32, device=self.device),
                                                           train)

        action = action.cpu().numpy()
        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray), 'Action dtype must be np.ndarray'
        return action

    @staticmethod
    def run_gradient_update_step(par_optimizer: Union[Actor, Critic], loss: torch.Tensor):
        """
        This function takes in an object containing trainable parameters and an optimizer,
        and using a given loss, runs one step of gradient update. If you set up trainable parameters
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        @param par_optimizer:
        @param loss:
        """
        par_optimizer.optimizer.zero_grad()
        loss.mean().backward()
        par_optimizer.optimizer.step()

    @staticmethod
    def critic_target_update(base_net: NeuralNetwork, target_net: NeuralNetwork,
                             tau: float, soft_update: bool):
        """
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        """
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        """
        This function represents one training iteration for the agent. It samples a batch
        from the replay buffer,and then updates the policy and critic networks
        using the sampled batch.
        """

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch

        # Compute targets for the Q-function
        with torch.no_grad():
            action, log_prob = self.actor.get_action_and_log_prob(s_prime_batch, deterministic=False)
            q1 = self.target_critic1.Q(torch.cat((s_prime_batch, action), dim=1))
            q2 = self.target_critic2.Q(torch.cat((s_prime_batch, action), dim=1))
            q = torch.min(q1, q2)
            q_target = r_batch + GAMMA * (q - self.alpha.get_param() * log_prob)

        # Update Q-functions by one step of gradient descent
        # Set everything to train mode
        self.critic1.Q.train()
        q1 = self.critic1.Q(torch.cat((s_batch, a_batch), dim=1))
        critic_loss1 = torch.nn.functional.mse_loss(q1, q_target)
        self.run_gradient_update_step(self.critic1, critic_loss1)

        self.critic2.Q.train()
        q2 = self.critic2.Q(torch.cat((s_batch, a_batch), dim=1))
        critic_loss2 = torch.nn.functional.mse_loss(q2, q_target)
        self.run_gradient_update_step(self.critic2, critic_loss2)

        # Update policy by one step of gradient ascent
        self.critic1.Q.eval()
        self.critic2.Q.eval()

        action, log_prob = self.actor.get_action_and_log_prob(s_batch, deterministic=False)
        q1 = self.critic1.Q(torch.cat((s_batch, action), dim=1))
        q2 = self.critic2.Q(torch.cat((s_batch, action), dim=1))
        q = torch.min(q1, q2)
        actor_loss = (self.alpha.get_param() * log_prob - q)
        self.run_gradient_update_step(self.actor, actor_loss)

        # Update alpha by one step of gradient ascent
        alpha_loss = -self.alpha.get_log_param() * (log_prob + self.target_entropy).detach()
        self.run_gradient_update_step(self.alpha, alpha_loss)

        # Soft update of the target networks
        self.critic_target_update(self.critic1.Q, self.target_critic1.Q, TAU, True)
        self.critic_target_update(self.critic2.Q, self.target_critic2.Q, TAU, True)


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evaluation episodes, or
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = True
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")

    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
