import torch
import torch.nn as nn

class PredictiveNetwork(nn.Module):
    """
    Recurrent neural network that predicts the plant's next state, given:
    - The current output: E.g. motor voltage
    - The previous state: E.g. motor velocity

    Will be trained with the following loss function:
    loss = (predicted_state - actual_state) ** 2
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(PredictiveNetwork, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden_state = None

    def forward(self, x):
        x, self.hidden_state = self.rnn(x, self.hidden_state)
        x = self.linear(x)
        return x
    
    def forward_once(self, x):
        x = x.unsqueeze(0).unsqueeze(0) # Add batch and sequence dimensions
        output, self.hidden_state = self.rnn(x, self.hidden_state)
        output = self.linear(output[:, -1, :]) # Get the last output in the sequence
        return output.squeeze(0) # Remove batch dimension
    
    def reset(self):
        self.hidden_state = None