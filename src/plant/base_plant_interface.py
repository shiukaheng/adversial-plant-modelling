from abc import ABC, abstractmethod

class BasePlantInterface(ABC):
    """
    The base plant interface that all plants must implement
    """

    @abstractmethod
    def set_output(self, output: float):
        """Sets the output of the plant

        Args:
            output (float): The output of the plant
        """

    @abstractmethod
    def measure_state(self, dt: float) -> float:
        """Measures the state of the plant

        Args:
            dt (float): The time step since the last measurement

        Returns:
            float: The state of the plant
        """

    def reset(self):
        """Resets the plant to its initial state
        """