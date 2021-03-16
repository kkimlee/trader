"""
Script for training Stock Trading Bot.

Usage:
  train.py <train-stock> <val-stock> <economy> [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug]

Options:
  --strategy=<strategy>             Q-learning strategy to use for training the network. Options:
                                      `dqn` i.e. Vanilla DQN,
                                      `t-dqn` i.e. DQN with fixed target distribution,
                                      `double-dqn` i.e. DQN with separate network for value estimation. [default: dqn]
  --window-size=<window-size>       Size of the n-day window stock data representation
                                    used as the feature vector. [default: 10]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch
                                    during training. [default: 32]
  --episode-count=<episode-count>   Number of trading episodes to use for training. [default: 1000]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --pretrained                      Specifies whether to continue training a previously
                                    trained model (reads `model-name`).
  --debug                           Specifies whether to use verbose logs during eval operation.
"""

import logging

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.utils import (
    get_stock_data,
    get_economy_data,
    format_currency,
    format_position,
    show_train_result,
    switch_k_backend_device
)

def main(train_stock, val_stock, economy, window_size, batch_size, ep_count,
         strategy="dqn", model_name="model_debug", pretrained=False,
         debug=False):
    """ Trains the stock trading bot using Deep Q-Learning.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python train.py --help]
    """
    print("initialize agent")
    agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)

    print('get stock data')
    train_data = get_stock_data(train_stock)
    print('get economy leading')
    economy_data = get_economy_data(economy)
    print('get val data')
    val_data = get_stock_data(val_stock)

    # 첫 째날과 둘 째 날의 종가의 차
    initial_offset = val_data[0][1] - val_data[0][0]
    last_checkpoint = 0

    for episode in range(1, ep_count + 1):
        print('train episode : ', episode)
        train_result, is_earlystopping = train_model(agent, episode, train_data, economy_data, ep_count=ep_count,
                                                     batch_size=batch_size, window_size=window_size, last_checkpoint=last_checkpoint)
        val_result, _ = evaluate_model(agent, val_data, economy_data, window_size, debug)
        show_train_result(train_result, val_result, initial_offset)

        if is_earlystopping == False:
            last_checkpoint = episode


if __name__ == "__main__":
    args = docopt(__doc__)

    train_stock = args["<train-stock>"]
    val_stock = args["<val-stock>"]
    economy_data = args["<economy>"]
    strategy = args["--strategy"]
    window_size = int(args["--window-size"])
    batch_size = int(args["--batch-size"])
    ep_count = int(args["--episode-count"])
    model_name = args["--model-name"]
    pretrained = args["--pretrained"]
    debug = args["--debug"]

    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    try:
        main(train_stock, val_stock, economy_data, window_size, batch_size,
             ep_count, strategy=strategy, model_name=model_name,
             pretrained=pretrained, debug=debug)
    except KeyboardInterrupt:
        print("Aborted!")
