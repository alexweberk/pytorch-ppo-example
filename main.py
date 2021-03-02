import gym

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

def kaiming_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        nn.init.constant_(layer.bias, 0)

class Memory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logprobs = []
        self.values = []

    def add(self, experience):
        state, action, reward, done, logprob, value = experience
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.logprobs.append(logprob)
        self.values.append(value)

class ActorCritic(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )
    def act(self, state):
        state = torch.tensor(state).float()
        with torch.no_grad():
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            logprob = dist.log_prob(action)
            state_value = self.critic(state)
            return action.item(), logprob, state_value

    def evaluate(self, states, actions):
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        logprobs = dist.log_prob(actions).view(-1, 1) # [update_timesteps, 1]
        state_values = self.critic(states) # [update_timesteps, 1]
        return logprobs, state_values, dist.entropy().view(-1, 1) # [update_timesteps, 1]

class PPOAgent:
    def __init__(self, n_state, n_action, n_hidden,
                    gamma=1.0, gae_lambda=0.95, lr=1e-3, eps_clip=0.2, k_epoch=4,
                    value_weight=0.5, entropy_weight=0.01):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epoch = k_epoch
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight

        self.model = ActorCritic(n_state, n_action, n_hidden)
        self.model.apply(kaiming_init)
        self.model_old = ActorCritic(n_state, n_action, n_hidden)
        self.model_old.load_state_dict(self.model.state_dict())

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.model.to(self.device)
        self.model_old.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = Memory()
        self.writer = SummaryWriter(comment=f'-ppo({eps_clip},{k_epoch},{value_weight},{entropy_weight})')

    def act(self, state):
        return self.model.act(state)

    def update(self):
        states_old = torch.tensor(self.memory.states).float().to(self.device)[:-1]
        actions_old = torch.tensor(self.memory.actions).to(self.device)[:-1]
        logprobs_old = torch.tensor(self.memory.logprobs).view(-1, 1).to(self.device)[:-1] # [update_timesteps, 1]

        rewards = self.memory.rewards[:-1]
        dones = self.memory.dones[:-1]
        values = self.memory.values
        returns = []
        gae = 0.0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * self.memory.values[i+1] * (1-dones[i]) - self.memory.values[i]
            gae = delta + self.gamma * self.gae_lambda * (1-dones[i]) * gae
            advantage = gae + values[i]
            returns.append(advantage)
        returns = torch.tensor(returns[::-1]).view(-1, 1).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # returns = []
        # discounted_reward = 0.0
        # for reward, done in zip(self.memory.rewards[::-1], self.memory.dones[::-1]):
        #     discounted_reward = reward + self.gamma * (1-done) * discounted_reward
        #     returns.append(discounted_reward)
        # returns = torch.FloatTensor(returns[::-1]).view(-1, 1).to(self.device) # [update_timesteps, 1]
        # returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        for _ in range(self.k_epoch):
            logprobs, state_values, entropies = self.model.evaluate(states_old, actions_old)

            assert returns.shape == state_values.shape == logprobs.shape == logprobs_old.shape == entropies.shape, \
                f"Shape mismatch: {returns.shape}, {state_values.shape}, {logprobs.shape}, {logprobs_old.shape}, {entropies.shape}."

            ratios = torch.exp(logprobs - logprobs_old)
            advantages = returns - state_values

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)

            policy_loss = -torch.min(surr1, surr2)
            value_loss = self.value_weight * self.criterion(returns, state_values)
            entropy_loss = -self.entropy_weight * entropies

            loss = policy_loss + value_loss + entropy_loss

            self.writer.add_scalar("Loss/Policy", policy_loss.mean().item(), self.episode)
            self.writer.add_scalar("Loss/Value", value_loss.mean().item(), self.episode)
            self.writer.add_scalar("Loss/Entropy", entropy_loss.mean().item(), self.episode)
            self.writer.add_scalar("Loss/Loss", loss.mean().item(), self.episode)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.model_old.load_state_dict(self.model.state_dict())



env = gym.make('CartPole-v0')
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_hidden = 64
gamma = 1.0
gae_lambda = 0.95
lr = 1e-3
eps_clip = 0.20
k_epoch = 4
value_weight = 0.5
entropy_weight = 0.01
update_timesteps = 300

agent = PPOAgent(n_state, n_action, n_hidden,
                gamma=gamma, gae_lambda=gae_lambda, lr=lr, eps_clip=eps_clip, k_epoch=k_epoch,
                value_weight=value_weight, entropy_weight=entropy_weight)

n_episode = 500

scores = [0] * n_episode
timestep = 0
for episode in range(n_episode):
    agent.episode = episode
    state = env.reset()
    while True:
        timestep += 1
        action, logprob, state_value = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        scores[episode] += reward

        agent.memory.add((state, action, reward, done, logprob, state_value))

        if timestep % update_timesteps == 0:
            agent.update()
            agent.memory.clear()
            timestep = 0

        if done:
            print(f"Episode {episode}, Reward {scores[episode]}")
            agent.writer.add_scalar("Reward/Reward", scores[episode], episode)
            break

        state = next_state

agent.writer.close()