from model.predictive_network import PredictiveNetwork
from model.adversial_controller_network import AdversialControllerNetwork
from plant.simulated_voltage_motor import SimulatedVoltageMotor
import torch
import torch.nn as nn
from model.maxmize_mse_loss import MaximizeMSELoss

class AdversialLearner:
    def __init__(self):
        self.predictive_network = PredictiveNetwork(2, 50, 1)
        self.adversial_controller_network = AdversialControllerNetwork(3, 50, 1)
        self.plant = SimulatedVoltageMotor(
            R=1.0, # Ohms
            L=0.5, # Henrys
            Kt=0.01, # Nm/A
            Ke=0.01, # Vs/rad
            J=0.01, # kgm^2
            B=0.01 # Nms
        )
        self.output = torch.zeros(1)
        self.last_state = torch.zeros(1)
        self.state = torch.zeros(1)
        self.predicted_state = torch.zeros(1)
        self.predictive_network_optimizer = torch.optim.Adam(self.predictive_network.parameters(), lr=0.001)
        self.adversial_controller_network_optimizer = torch.optim.Adam(self.adversial_controller_network.parameters(), lr=0.001)
        self.predictive_network_loss = nn.MSELoss()
        self.adversial_controller_network_loss = MaximizeMSELoss()
    def update(self):
        # Generate a new output
        self.output = self.adversial_controller_network.forward_once(torch.cat((self.output, self.state, self.predicted_state)))
        # Apply the output to the plant and measure the new state
        self.plant.set_output(self.output.item())
        self.last_state = self.state
        self.state = torch.tensor([self.plant.measure_state(0.01)])
        # Predict the new state
        self.predictive_network.zero_grad()
        self.predicted_state = self.predictive_network.forward_once(torch.cat((self.output, self.last_state)))
        # Compute the loss
        loss = self.predictive_network_loss(self.predicted_state, self.state)
        # Backpropagate
        loss.backward()
        self.predictive_network_optimizer.step()
