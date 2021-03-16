from torch import optim
import torch.nn as nn

class Brain(object):
    def __init__(self, actor_critic, num_steps, num_processes):
        self.actor_critic = actor_critic  # actor_critic은 Net 클래스로 구현한 신경망
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.01)
        self.NUM_ADVANCED_STEP = num_steps
        self.NUM_PROCESSES = num_processes

         # A2C 손실함수 계산에 사용되는 상수
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

    def update(self, rollouts):
        '''Advantage학습의 대상이 되는 5단계 모두를 사용하여 수정'''
        obs_shape = rollouts.observations.size()[2:]  # torch.Size([4, 84, 84])
        num_steps = self.NUM_ADVANCED_STEP
        num_processes = self.NUM_PROCESSES

        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, 4),
            rollouts.actions.view(-1, 1))

        # 주의 : 각 변수의 크기
        # rollouts.observations[:-1].view(-1, 4) torch.Size([80, 4])
        # rollouts.actions.view(-1, 1) torch.Size([80, 1])
        # values torch.Size([80, 1])
        # action_log_probs torch.Size([80, 1])
        # entropy torch.Size([])

        values = values.view(num_steps, num_processes,
                             1)  # torch.Size([5, 16, 1])
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # advantage(행동가치-상태가치) 계산
        advantages = rollouts.returns[:-1] - values  # torch.Size([5, 16, 1])

        # Critic의 loss 계산
        value_loss = advantages.pow(2).mean()

        # Actor의 gain 계산, 나중에 -1을 곱하면 loss가 된다
        action_gain = (action_log_probs*advantages.detach()).mean()
        # detach 메서드를 호출하여 advantages를 상수로 취급

        # 오차함수의 총합
        total_loss = (value_loss * self.value_loss_coef -
                      action_gain - entropy * self.entropy_coef)

        # 결합 가중치 수정
        self.actor_critic.train()  # 신경망을 학습 모드로 전환
        self.optimizer.zero_grad()  # 경사를 초기화
        total_loss.backward()  # 역전파 계산
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        # 결합 가중치가 한번에 너무 크게 변화하지 않도록, 경사를 0.5 이하로 제한함(클리핑)

        self.optimizer.step()  # 결합 가중치 수정
