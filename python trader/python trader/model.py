# A2C에 사용되는 신경망 구성
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.actor = nn.Linear(n_mid, n_out)  # action의 확률 출력
        self.critic = nn.Linear(n_mid, 1)  # state의 가치 출력

    def forward(self, x):
        '''신경망 순전파 계산을 정의'''
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        critic_output = self.critic(h2)  # 상태가치 
        actor_output = self.actor(h2)  # 행동 계산

        return critic_output, actor_output

    def act(self, x):
        '''상태 x로부터 행동을 확률적으로 결정'''
        value, actor_output = self(x)
        # dim=1이므로 행동의 종류에 대해 softmax를 적용
        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(num_samples=1)  # dim=1이므로 행동의 종류에 대해 확률을 계산
        return action

    def get_value(self, x):
        '''상태 x로부터 상태가치를 계산'''
        value, actor_output = self(x)

        return value

    def evaluate_actions(self, x, actions):
        '''상태 x로부터 상태가치, 실제 행동 actions의 로그 확률, 엔트로피를 계산'''
        value, actor_output = self(x)

        log_probs = F.log_softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류에 대해 확률을 계산
        action_log_probs = log_probs.gather(1, actions)  # 실제 행동의 로그 확률(log_probs)을 구함

        probs = F.softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류에 대한 계산
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy
