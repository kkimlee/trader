import random
import numpy as np

from collections import deque

import torch
import torch.nn as nn
from torch import optim



class Agent:

    # 초기화
    def __init__(self, train = True):

        self.input_size = 10 # 입력 데이터 개수
        self.action_size = 2 # 행동 매수,유지 or 매도,유지

        # 각 모델의 행동 결과 기록
        self.buy_memory = deque(maxlen=10000) # 매수 기록
        self.sell_memory = deque(maxlen=10000) # 매도 기록

        # 각 모델의 랜덤 선택 확률
        self.buy_epsilon = 1.0 # buy_model 매수 행동 확률
        self.sell_epsilon = 1.0 # sell_model 매도 행동 확률

        self.epsilon_decay = 0.995 # 랜덤 확률 감소 비율
       

        

        # 학습 모드
        if train:
            self.buy_model = model()
            self.sell_model = model()

    # 모델 생성
    def model(self):
        model = nn.Sequential(
            nn.Linear(self.input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.action_size)
            )

        return model

    # 매수 결과 저장
    def buy_remember(self, state, action, reward, next_state, done):

        self.buy_memory.append((state, action, reward, next_state, done))

    # 매도 결과 저장
    def sell_remember(self, state, action, rewrad, next_state, done):

        self.sell_memory.append(state, action, reward, next_state, done)

    # 매수 모델 행동
    def buy_act(self, state):
        
        if random.random() <= self.buy_epsilon:
            return random.randrange(self.action_size)

        else:
            profit = self.buy_model(state)
            return np.argmax(profit[0])

    # 매도 모델 행동
    def sell_act(self, state):

        if random.random() <= self.sell_epsilon:
            return random.randrange(self.action_size)

        else:
            profit = self.sell_model(state)
            return np.argmax(profit[0])
        


