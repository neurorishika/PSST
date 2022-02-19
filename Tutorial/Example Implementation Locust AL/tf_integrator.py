import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


def tf_check_type(t, y0):  # Ensure Input is Correct
    """
    This function checks the type of the input to ensure that it is a floating point number.
    """
    if not (y0.dtype.is_floating and t.dtype.is_floating):
        raise TypeError("Error: y0 and t must be floating point numbers.")


class _Tf_Integrator:
    """
    This class implements the Runge-Kutta 4th order method in TensorFlow with last Spike Time Calculation.
    """

    def __init__(self, n_, F_b):
        """
        This function initializes the class with the number of neurons and the firing threshold.

        Parameters
        ----------
        n_ : int
            Number of neurons in the system.
        F_b : list or np.array of floats
            Firing threshold of the neurons.
        """
        # class constructor to get inputs for number of neurons and firing thresholds
        self.n_ = n_
        self.F_b = F_b

    def integrate(self, func, y0, t):
        """
        This function integrates a function func using the Runge-Kutta 4th order method in TensorFlow with last Spike Time Calculation.

        Parameters:
        -----------
        func: function
            The function to be integrated.
        y0: float
            The initial condition.
        t: numpy array
            The time array.
        """
        time_delta_grid = t[1:] - t[:-1]  # define the time step at each point

        def scan_func(y, t_dt):
            """
            This function performs the integration step and last spike time calculation.
            
            Parameters:
            -----------
            y: float
                The value of y at which the function is being evaluated.
            t_dt: (float, float)
                The time point and time step at which the function is being evaluated.
            """
            # recall the necessary variables
            n_ = self.n_  # number of neurons
            F_b = self.F_b  # firing threshold

            t, dt = t_dt  # unpack the time point and time step

            # Differential updation
            dy = self._step_func(func, t, dt, y)  # Make code more modular.
            dy = tf.cast(dy, dtype=y.dtype)  # Failsafe

            out = y + dy  # the result after differential updation

            # Conditional to use specialized Integrator vs Normal Integrator (n=0)
            if n_ > 0:

                # Extract the last n variables for fire times
                fire_t = y[-n_:]

                # Value of change in firing times if neuron didnt fire = 0
                l = tf.zeros(tf.shape(fire_t), dtype=fire_t.dtype)

                # Value of change in firing times if neuron fired = Current Time - Last Fire Time
                l_ = t - fire_t

                # Check if Voltage is initially less than Firing Threshold
                z = tf.less(y[:n_], F_b)
                # Check if Voltage is more than Firing Threshold after updation
                z_ = tf.greater_equal(out[:n_], F_b)

                df = tf.where(tf.logical_and(z, z_), l_, l)

                fire_t_ = fire_t + df  # Update firing time

                return tf.concat(
                    [out[:-n_], fire_t_], 0
                )  # Remove and Update the last n variables with new last spike times
            else:
                return out

        y = tf.scan(scan_func, (t[:-1], time_delta_grid), y0)  # Perform the integration

        return tf.concat([[y0], y], axis=0)  # Add the initial condition to the result and return

    def _step_func(self, func, t, dt, y):
        """
        This function determines the size of the step.

        Parameters:
        -----------
        func: function
            The function to be integrated.
        t: float
            The time point at which the function is being evaluated.
        dt: float
            The time step at which the function is being integrated.
        y: float
            The value of y at which the function is being evaluated.
        """
        k1 = func(y, t)
        half_step = t + dt / 2
        dt_cast = tf.cast(dt, y.dtype)  # Failsafe

        k2 = func(y + dt_cast * k1 / 2, half_step)
        k3 = func(y + dt_cast * k2 / 2, half_step)
        k4 = func(y + dt_cast * k3, t + dt)
        return tf.add_n([k1, 2 * k2, 2 * k3, k4]) * (dt_cast / 6)


def odeint(func, y0, t, n_, F_b):
    """
    This function integrates the function func using the modified Runge-Kutta 4th order method implemented in the _Tf_Integrator class

    Parameters:
    -----------
    func: function
        The function to be integrated.
    y0: float
        The initial condition.
    t: numpy array
        The time array.
    n_: int
        Number of neurons in the system.
    F_b: list or np.array of floats
        Firing threshold of the neurons.
    """
    # Ensure Input is in the form of TensorFlow Tensors
    t = tf.convert_to_tensor(t, preferred_dtype=tf.float64, name="t")
    y0 = tf.convert_to_tensor(y0, name="y0")
    tf_check_type(y0, t)  # Ensure Input is of the correct type
    return _Tf_Integrator(n_, F_b).integrate(func, y0, t)

