import torch
import torch.nn as nn

class AdversialControllerNetwork(nn.Module):
    """
    Recurrent neural network that generates the plant's next output, given:
    - The current output: E.g. motor voltage
    - The current state: E.g. motor velocity
    - The predicted current state: E.g. motor velocity

    The goal of this network is to generate an output that maximizes the error between the predicted state and the actual state.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(AdversialControllerNetwork, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden_state = None

    def forward(self, x):
        x, self.hidden_state = self.rnn(x, self.hidden_state)
        x = self.linear(x)
        return x
    
    def reset(self):
        self.hidden_state = None