import gym
import numpy as np

import torch

from model import Net
from brain import Brain
from rolloutstorage import RolloutStorage

'''
기존 A2C가 망한이유
1. keras는 Gradient 학습이 어려움
2. A2C는 DQN 방식과 달리 Replay Buffer를 사용하지 않음 따라서 잘못 학습되면 그냥 잘못된 걸로 계속 학습함 -> 개선이 안됨
3. Cartpole이나 Froze Lake 등의 간단한 환경에서는 결과가 목표를 달성했느냐 못했느냐 O, X 문제였기 때문에 단일 환경에서도 문제가 없었음
4. Trader의 목표는 매우 작은 이익을 내는 것이 아니라 최대의 이익을 얻는 것. \
   그러나 단일 환경의 A2C Trader는 이익을 얻었다가 손해를 보는 경험을 한경우 이익을 얻은 후 손해를 보지 않기 위해 투자를 하지 않음
5. 즉, A2C의 핵심이라 할 수 있는 다양한 환경에서 경험을 하지 않음
6. 사실상 구현한 것은 Synchronous Advantage Actor-Critic (A2C)가 아닌 Actor critic 이였기 때문에 성능이 안나오는 것이 당연함

7. 기존의 trader를 개선할 방법은 2가지임 
    - Actor-Critic with Experience Replay(ACER)로 구현
    - 제대로된 A2C 등의 최신 기술을 적용

trader 환경 만드는 중!

'''

class Environment:
    def __init__(self):
        # 상수 정의(원본)
        self.ENV = 'CartPole-v0'  # 태스크 이름
        self.NUM_EPISODES = 1000  # 최대 에피소드 수

        self.NUM_PROCESSES = 32  # 동시 실행 환경 수
        self.NUM_ADVANCED_STEP = 5  # 총 보상을 계산할 때 Advantage 학습을 할 단계 수

        # Agent 정보
        self.state_size = 10
        self.action_size = 2

    def run(self):
        '''실행 엔트리 포인트'''

        # 동시 실행할 환경 수 만큼 env를 생성
        envs = [gym.make(self.ENV) for i in range(self.NUM_PROCESSES)]

        # 모든 에이전트가 공유하는 Brain 객체를 생성
        n_in = envs[0].observation_space.shape[0]  # 상태 변수 수는 4
        n_out = envs[0].action_space.n  # 행동 가짓수는 2
        n_mid = 32
        actor_critic = Net(n_in, n_mid, n_out)  # 신경망 객체 생성
        global_brain = Brain(actor_critic, self.NUM_ADVANCED_STEP, self.NUM_PROCESSES)

        # 매수 모델
        buy_actor_critic = Net(self.state_size, 32, self.action_size) # 신경망 객체
        buy_global_brain = Brain(buy_actor_critic, self.NUM_ADVANCED_STEP, self.NUM_PROCESSES)

        # 매도 모델
        sell_actor_critic = Net(self.state_size, 32, self.action_size) # 신경망 객체
        buy_global_brain = Brain(sell_actor_critic, self.NUM_ADVANCED_STEP, self.NUM_PROCESSES)
        

        # 각종 정보를 저장하는 변수
        obs_shape = n_in
        current_obs = torch.zeros(
            self.NUM_PROCESSES, obs_shape)  # torch.Size([16, 4])
        rollouts = RolloutStorage(
            self.NUM_ADVANCED_STEP, self.NUM_PROCESSES, obs_shape)  # rollouts 객체
        episode_rewards = torch.zeros([self.NUM_PROCESSES, 1])  # 현재 에피소드의 보상
        final_rewards = torch.zeros([self.NUM_PROCESSES, 1])  # 마지막 에피소드의 보상
        obs_np = np.zeros([self.NUM_PROCESSES, obs_shape])  # Numpy 배열
        reward_np = np.zeros([self.NUM_PROCESSES, 1])  # Numpy 배열
        done_np = np.zeros([self.NUM_PROCESSES, 1])  # Numpy 배열
        each_step = np.zeros(self.NUM_PROCESSES)  # 각 환경의 단계 수를 기록
        episode = 0  # 환경 0의 에피소드 수

        # 초기 상태로부터 시작
        obs = [envs[i].reset() for i in range(self.NUM_PROCESSES)]
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float()  # torch.Size([16, 4])
        current_obs = obs  # 가장 최근의 obs를 저장
        
        # advanced 학습에 사용되는 객체 rollouts 첫번째 상태에 현재 상태를 저장
        rollouts.observations[0].copy_(current_obs)

        # 1 에피소드에 해당하는 반복문
        for j in range(self.NUM_EPISODES*self.NUM_PROCESSES):  # 전체 for문
            # advanced 학습 대상이 되는 각 단계에 대해 계산
            for step in range(self.NUM_ADVANCED_STEP):

                # 행동을 선택
                with torch.no_grad():
                    action = actor_critic.act(rollouts.observations[step])

                # (16,1)→(16,) -> tensor를 NumPy변수로
                actions = action.squeeze(1).numpy()

                # 한 단계를 실행
                for i in range(self.NUM_PROCESSES):
                    obs_np[i], reward_np[i], done_np[i], _ = envs[i].step(
                        actions[i])

                    # episode의 종료가치, state_next를 설정
                    if done_np[i]:  # 단계 수가 200을 넘거나, 봉이 일정 각도 이상 기울면 done이 True가 됨

                        # 환경 0일 경우에만 출력
                        if i == 0:
                            print('%d Episode: Finished after %d steps' % (
                                episode, each_step[i]+1))
                            episode += 1

                        # 보상 부여
                        if each_step[i] < 195:
                            reward_np[i] = -1.0  # 도중에 봉이 넘어지면 페널티로 보상 -1 부여
                        else:
                            reward_np[i] = 1.0  # 봉이 쓰러지지 않고 끝나면 보상 1 부여

                        each_step[i] = 0  # 단계 수 초기화
                        obs_np[i] = envs[i].reset()  # 실행 환경 초기화

                    else:
                        reward_np[i] = 0.0  # 그 외의 경우는 보상 0 부여
                        each_step[i] += 1

                # 보상을 tensor로 변환하고, 에피소드의 총보상에 더해줌
                reward = torch.from_numpy(reward_np).float()
                episode_rewards += reward

                # 각 실행 환경을 확인하여 done이 true이면 mask를 0으로, false이면 mask를 1로
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done_np])

                # 마지막 에피소드의 총 보상을 업데이트
                final_rewards *= masks  # done이 false이면 1을 곱하고, true이면 0을 곱해 초기화
                # done이 false이면 0을 더하고, true이면 episode_rewards를 더해줌
                final_rewards += (1 - masks) * episode_rewards

                # 에피소드의 총보상을 업데이트
                episode_rewards *= masks  # done이 false인 에피소드의 mask는 1이므로 그대로, true이면 0이 됨

                # 현재 done이 true이면 모두 0으로 
                current_obs *= masks

                # current_obs를 업데이트
                obs = torch.from_numpy(obs_np).float()  # torch.Size([16, 4])
                current_obs = obs  # 최신 상태의 obs를 저장

                # 메모리 객체에 현 단계의 transition을 저장
                rollouts.insert(current_obs, action.data, reward, masks)

            # advanced 학습 for문 끝

            # advanced 학습 대상 중 마지막 단계의 상태로 예측하는 상태가치를 계산

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.observations[-1]).detach()
                # rollouts.observations의 크기는 torch.Size([6, 16, 4])

            # 모든 단계의 할인총보상을 계산하고, rollouts의 변수 returns를 업데이트
            rollouts.compute_returns(next_value)

            # 신경망 및 rollout 업데이트
            global_brain.update(rollouts)
            rollouts.after_update()

            # 환경 갯수를 넘어서는 횟수로 200단계를 버텨내면 성공
            if final_rewards.sum().numpy() >= self.NUM_PROCESSES:
                print('연속성공')
                break
