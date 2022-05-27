import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
import tf_integrator as tf_int
import sys

n_n = 3  # Number of simultaneous neurons to simulate

sim_res = 0.01  # Time Resolution of the Simulation
sim_time = 700  # Length of the Simulation

# t = np.arange(0,sim_time,sim_res)
t = np.load("time.npy", allow_pickle=True)[int(sys.argv[1])]  # get first argument to run.py

# Acetylcholine

if sys.argv[1] == "0":
    ach_mat = np.zeros((n_n, n_n))  # Ach Synapse Connectivity Matrix
    ach_mat[1, 0] = 1  # If connectivity is random, once initialized it will be the same.
    np.save("ach_mat", ach_mat)
else:
    ach_mat = np.load("ach_mat.npy")

## PARAMETERS FOR ACETYLCHLOLINE SYNAPSES ##

n_ach = int(np.sum(ach_mat))  # Number of Acetylcholine (Ach) Synapses
alp_ach = [10.0] * n_ach  # Alpha for Ach Synapse
bet_ach = [0.2] * n_ach  # Beta for Ach Synapse
t_max = 0.3  # Maximum Time for Synapse
t_delay = 0  # Axonal Transmission Delay
A = [0.5] * n_n  # Synaptic Response Strength
g_ach = [0.35] * n_n  # Ach Conductance
E_ach = [0.0] * n_n  # Ach Potential

# GABAa

if sys.argv[1] == "0":
    gaba_mat = np.zeros((n_n, n_n))  # GABAa Synapse Connectivity Matrix
    gaba_mat[2, 1] = 1  # If connectivity is random, once initialized it will be the same.
    np.save("gaba_mat", gaba_mat)
else:
    gaba_mat = np.load("gaba_mat.npy")
## PARAMETERS FOR GABAa SYNAPSES ##

n_gaba = int(np.sum(gaba_mat))  # Number of GABAa Synapses
alp_gaba = [10.0] * n_gaba  # Alpha for GABAa Synapse
bet_gaba = [0.16] * n_gaba  # Beta for GABAa Synapse
V0 = [-20.0] * n_n  # Decay Potential
sigma = [1.5] * n_n  # Decay Time Constant
g_gaba = [0.8] * n_n  # fGABA Conductance
E_gaba = [-70.0] * n_n  # fGABA Potential

## Storing Firing Thresholds ##
F_b = [0.0] * n_n  # Fire threshold


def I_inj_t(t):
    """
    This function returns the external current to be injected into the network at any time step from the current_input matrix.

    Parameters:
    -----------
    t: float
        The time at which the current injection is being performed.
    """
    # Turn indices to integer and extract from matrix
    index = tf.cast(t / sim_res, tf.int32)
    return tf.constant(current_input.T, dtype=tf.float64)[index]


## Acetylcholine Synaptic Current ##


def I_ach(o, V):
    """
    This function returns the synaptic current for the Acetylcholine (Ach) synapses for each neuron.

    Parameters:
    -----------
    o: float
        The fraction of open acetylcholine channels for each synapse.
    V: float
        The membrane potential of the postsynaptic neuron.
    """
    o_ = tf.constant(
        [0.0] * n_n ** 2, dtype=tf.float64
    )  # Initialize the flattened matrix to store the synaptic open fractions
    ind = tf.boolean_mask(tf.range(n_n ** 2), ach_mat.reshape(-1) == 1)  # Get the indices of the synapses that exist
    o_ = tf.tensor_scatter_nd_update(o_, tf.reshape(ind, [-1, 1]), o)  # Update the flattened open fraction matrix
    o_ = tf.transpose(
        tf.reshape(o_, (n_n, n_n))
    )  # Reshape and Transpose the matrix to be able to multiply it with the conductance matrix
    return tf.reduce_sum(tf.transpose((o_ * (V - E_ach)) * g_ach), 1)  # Calculate the synaptic current


## GABAa Synaptic Current ##


def I_gaba(o, V):
    """
    This function returns the synaptic current for the GABA synapses for each neuron.

    Parameters:
    -----------
    o: float
        The fraction of open GABA channels for each synapse.
    V: float
        The membrane potential of the postsynaptic neuron.
    """
    o_ = tf.constant(
        [0.0] * n_n ** 2, dtype=tf.float64
    )  # Initialize the flattened matrix to store the synaptic open fractions
    ind = tf.boolean_mask(tf.range(n_n ** 2), gaba_mat.reshape(-1) == 1)  # Get the indices of the synapses that exist
    o_ = tf.tensor_scatter_nd_update(o_, tf.reshape(ind, [-1, 1]), o)  # Update the flattened open fraction matrix
    o_ = tf.transpose(
        tf.reshape(o_, (n_n, n_n))
    )  # Reshape and Transpose the matrix to be able to multiply it with the conductance matrix
    return tf.reduce_sum(tf.transpose((o_ * (V - E_gaba)) * g_gaba), 1)  # Calculate the synaptic current


## Other Currents ##


def I_K(V, n):
    """
    This function determines the K-channel current.

    Parameters:
    -----------
    V: float
        The membrane potential.
    n: float 
        The K-channel gating variable n.
    """
    return g_K * n ** 4 * (V - E_K)


def I_Na(V, m, h):
    """
    This function determines the Na-channel current.
    
    Parameters:
    -----------
    V: float
        The membrane potential.
    m: float
        The Na-channel gating variable m.
    h: float
        The Na-channel gating variable h.
    """
    return g_Na * m ** 3 * h * (V - E_Na)


def I_L(V):
    """
    This function determines the leak current.

    Parameters:
    -----------
    V: float
        The membrane potential.
    """
    return g_L * (V - E_L)


def dXdt(X, t):
    """
    This function determines the derivatives of the membrane voltage and gating variables for n_n neurons.

    Parameters:
    -----------
    X: float
        The state vector given by the [V1,V2,...,Vn_n,m1,m2,...,mn_n,h1,h2,...,hn_n,n1,n2,...,nn_n] where 
            Vx is the membrane potential for neuron x
            mx is the Na-channel gating variable for neuron x 
            hx is the Na-channel gating variable for neuron x
            nx is the K-channel gating variable for neuron x.
    t: float
        The time points at which the derivatives are being evaluated.
    """
    V = X[: 1 * n_n]  # First n_n values are Membrane Voltage
    m = X[1 * n_n : 2 * n_n]  # Next n_n values are Sodium Activation Gating Variables
    h = X[2 * n_n : 3 * n_n]  # Next n_n values are Sodium Inactivation Gating Variables
    n = X[3 * n_n : 4 * n_n]  # Next n_n values are Potassium Gating Variables
    o_ach = X[4 * n_n : 4 * n_n + n_ach]  # Next n_ach values are Acetylcholine Synapse Open Fractions
    o_gaba = X[4 * n_n + n_ach : 4 * n_n + n_ach + n_gaba]  # Next n_gaba values are GABAa Synapse Open Fractions
    fire_t = X[-n_n:]  # Last n_n values are the last fire times as updated by the modified integrator

    dVdt = (
        I_inj_t(t) - I_Na(V, m, h) - I_K(V, n) - I_L(V) - I_ach(o_ach, V) - I_gaba(o_gaba, V)
    ) / C_m  # The derivative of the membrane potential

    ## Updation for gating variables ##

    m0, tm, h0, th = Na_prop(V)  # Calculate the dynamics of the Na-channel gating variables for all n_n neurons
    n0, tn = K_prop(V)  # Calculate the dynamics of the K-channel gating variables for all n_n neurons

    dmdt = -(1.0 / tm) * (m - m0)  # The derivative of the Na-channel gating variable m for all n_n neurons
    dhdt = -(1.0 / th) * (h - h0)  # The derivative of the Na-channel gating variable h for all n_n neurons
    dndt = -(1.0 / tn) * (n - n0)  # The derivative of the K-channel gating variable n for all n_n neurons

    ## Updation for o_ach ##

    A_ = tf.constant(A, dtype=tf.float64)  # Get the synaptic response strengths of the pre-synaptic neurons
    Z_ = tf.zeros(tf.shape(A_), dtype=tf.float64)  # Create a zero matrix of the same size as A_

    T_ach = tf.where(
        tf.logical_and(tf.greater(t, fire_t + t_delay), tf.less(t, fire_t + t_max + t_delay)), A_, Z_
    )  # Find which synapses would have received an presynaptic spike in the past window and assign them the corresponding synaptic response strength
    T_ach = tf.multiply(
        tf.constant(ach_mat, dtype=tf.float64), T_ach
    )  # Find the postsynaptic neurons that would have received an presynaptic spike in the past window
    T_ach = tf.boolean_mask(
        tf.reshape(T_ach, (-1,)), ach_mat.reshape(-1) == 1
    )  # Get the pre-synaptic activation function for only the existing synapses

    do_achdt = (
        alp_ach * (1.0 - o_ach) * T_ach - bet_ach * o_ach
    )  # Calculate the derivative of the open fraction of the acetylcholine synapses

    ## Updation for o_gaba ##

    T_gaba = 1.0 / (
        1.0 + tf.exp(-(V - V0) / sigma)
    )  # Calculate the presynaptic activation function for all n_n neurons
    T_gaba = tf.multiply(
        tf.constant(gaba_mat, dtype=tf.float64), T_gaba
    )  # Find the postsynaptic neurons that would have received an presynaptic spike in the past window
    T_gaba = tf.boolean_mask(
        tf.reshape(T_gaba, (-1,)), gaba_mat.reshape(-1) == 1
    )  # Get the pre-synaptic activation function for only the existing synapses

    do_gabadt = (
        alp_gaba * (1.0 - o_gaba) * T_gaba - bet_gaba * o_gaba
    )  # Calculate the derivative of the open fraction of the GABAa synapses

    ## Updation for fire times ##

    dfdt = tf.zeros(
        tf.shape(fire_t), dtype=fire_t.dtype
    )  # zero change in fire_t as it will be updated by the modified integrator

    out = tf.concat(
        [dVdt, dmdt, dhdt, dndt, do_achdt, do_gabadt, dfdt], 0
    )  # Concatenate the derivatives of the membrane potential, gating variables, and open fractions
    return out


def K_prop(V):
    """
    This function determines the K-channel gating dynamics.

    Parameters:
    -----------
    V: float
        The membrane potential.
    """
    T = 22  # Temperature
    phi = 3.0 ** ((T - 36.0) / 10)  # Temperature-correction factor
    V_ = V - (-50)  # Voltage baseline shift

    alpha_n = 0.02 * (15.0 - V_) / (tf.exp((15.0 - V_) / 5.0) - 1.0)  # Alpha for the K-channel gating variable n
    beta_n = 0.5 * tf.exp((10.0 - V_) / 40.0)  # Beta for the K-channel gating variable n

    t_n = 1.0 / ((alpha_n + beta_n) * phi)  # Time constant for the K-channel gating variable n
    n_0 = alpha_n / (alpha_n + beta_n)  # Steady-state value for the K-channel gating variable n

    return n_0, t_n


def Na_prop(V):
    """
    This function determines the Na-channel gating dynamics.
    
    Parameters:
    -----------
    V: float
        The membrane potential.
    """
    T = 22  # Temperature
    phi = 3.0 ** ((T - 36) / 10)  # Temperature-correction factor
    V_ = V - (-50)  # Voltage baseline shift

    alpha_m = 0.32 * (13.0 - V_) / (tf.exp((13.0 - V_) / 4.0) - 1.0)  # Alpha for the Na-channel gating variable m
    beta_m = 0.28 * (V_ - 40.0) / (tf.exp((V_ - 40.0) / 5.0) - 1.0)  # Beta for the Na-channel gating variable m

    alpha_h = 0.128 * tf.exp((17.0 - V_) / 18.0)  # Alpha for the Na-channel gating variable h
    beta_h = 4.0 / (tf.exp((40.0 - V_) / 5.0) + 1.0)  # Beta for the Na-channel gating variable h

    t_m = 1.0 / ((alpha_m + beta_m) * phi)  # Time constant for the Na-channel gating variable m
    t_h = 1.0 / ((alpha_h + beta_h) * phi)  # Time constant for the Na-channel gating variable h

    m_0 = alpha_m / (alpha_m + beta_m)  # Steady-state value for the Na-channel gating variable m
    h_0 = alpha_h / (alpha_h + beta_h)  # Steady-state value for the Na-channel gating variable h

    return m_0, t_m, h_0, t_h


# Initializing the Parameters

C_m = [1.0] * n_n  # Membrane capacitances
g_K = [10.0] * n_n  # K-channel conductances
E_K = [-95.0] * n_n  # K-channel reversal potentials

g_Na = [100] * n_n  # Na-channel conductances
E_Na = [50] * n_n  # Na-channel reversal potentials

g_L = [0.15] * n_n  # Leak conductances
E_L = [-55.0] * n_n  # Leak reversal potentials

# Creating the Current Input

if sys.argv[1] == "0":
    current_input = np.zeros((n_n, int(sim_time / sim_res)))  # The current input to the network
    current_input[0, int(100 / sim_res) : int(200 / sim_res)] = 2.5
    current_input[0, int(300 / sim_res) : int(400 / sim_res)] = 5.0
    current_input[0, int(500 / sim_res) : int(600 / sim_res)] = 7.5
    np.save("current_input", current_input)
else:
    current_input = np.load("current_input.npy")

# State Vector Definition #

if sys.argv[1] == "0":
    state_vector = [-71] * n_n + [0, 0, 0] * n_n + [0] * n_ach + [0] * n_gaba + [-9999999] * n_n
    state_vector = np.array(state_vector)
    state_vector = state_vector + 0.01 * state_vector * np.random.normal(size=state_vector.shape)
    np.save("state_vector", state_vector)
else:
    state_vector = np.load("state_vector.npy")


# Define the Number of Batches
n_batch = 2

# Split t array into batches using numpy
t_batch = np.array_split(t, n_batch)

# Iterate over the batches of time array
for n, i in enumerate(t_batch):

    # Inform start of Batch Computation
    print("Batch", (n + 1), "Running...", end="")

    # In np.array_split(), the split edges are present in only one array and since
    # our initial vector to successive calls is corresposnding to the last output
    # our first element in the later time array should be the last element of the
    # previous output series, Thus, we append the last time to the beginning of
    # the current time array batch.
    if n > 0:
        i = np.append(i[0] - sim_res, i)

    # Set state_vector as the initial condition
    init_state = tf.constant(state_vector, dtype=tf.float64)
    # Create the Integrator computation graph over the current batch of t array
    tensor_state = tf_int.odeint(dXdt, init_state, i, n_n, F_b)

    # Initialize variables and run session
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        state = sess.run(tensor_state)
        sess.close()

    # Reset state_vector as the last element of output
    state_vector = state[-1, :]

    # Save the output of the simulation to a binary file
    np.save("batch" + str(int(sys.argv[1]) + 1) + "_part_" + str(n + 1), state)

    # Clear output
    state = None

    print("Finished")
