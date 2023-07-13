from plant.base_plant_interface import BasePlantInterface

class SimulatedVoltageMotor(BasePlantInterface):
    def __init__(self, R, L, Kt, Ke, J, B):
        self.R = R
        self.L = L
        self.Kt = Kt
        self.Ke = Ke
        self.J = J
        self.B = B
        self.voltage = 0.0
        self.current = 0.0
        self.omega = 0.0

    def set_output(self, output: float):
        self.voltage = output

    def measure_state(self, dt: float) -> float:
        # Use Euler's method to approximate the solution to the differential equations
        di_dt = (self.voltage - self.R * self.current - self.Ke * self.omega) / self.L
        domega_dt = (self.Kt * self.current - self.B * self.omega) / self.J

        self.current += di_dt * dt
        self.omega += domega_dt * dt

        return self.omega

    def reset(self):
        self.voltage = 0.0
        self.current = 0.0
        self.omega = 0.0