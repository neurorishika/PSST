import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def tf_check_type(t, y0): # Ensure Input is Correct
    if not (y0.dtype.is_floating and t.dtype.is_floating):
        raise TypeError('Error in Datatype')

def tf_check_increasing(t): # Ensure Time is Monotonically Increasing
    assert_op = tf.Assert(tf.reduce_all(t[1:]>t[:-1]),["Time must be monotonic"])
    return tf.control_dependencies([assert_op])

class _Tf_Integrator():
    
    def __init__(self,n_,F_b): 
        # class constructor to get inputs for number of neurons and firing thresholds
        self.n_ = n_
        self.F_b = F_b
    
    def integrate(self, func, y0, t): 
        time_delta_grid = t[1:] - t[:-1]
        scan_func = self._make_scan_func(func)
        y = tf.scan(scan_func, (t[:-1], time_delta_grid),y0)
        return tf.concat([[y0], y], axis=0)
    
    def _make_scan_func(self, func): # stepper function
        def scan_func(y, t_dt): 
            # recall the necessary variables
            n_ = self.n_
            F_b = self.F_b
            
            t, dt = t_dt
            
            # Differential updation
            dy = self._step_func(func,t,dt,y) # Make code more modular.
            dy = tf.cast(dy, dtype=y.dtype) # Failsafe
           
            out = y + dy # the result after differential updation
        
            # Conditional to use specialized Integrator vs Normal Integrator (n=0)
            if n_>0:
                
                # Extract the last n variables for fire times
                fire_t = y[-n_:] 
                
                # Value of change in firing times if neuron didnt fire = 0
                l = tf.zeros(tf.shape(fire_t),dtype=fire_t.dtype) 
                
                # Value of change in firing times if neuron fired = Current Time - Last Fire Time
                l_ = t-fire_t 
                
                # Check if Voltage is initially less than Firing Threshold
                z = tf.less(y[:n_],F_b)              
                # Check if Voltage is more than Firing Threshold after updation
                z_ = tf.greater_equal(out[:n_],F_b)  
                
                # tf.where(cond,a,b) chooses elements from a/b based on condition 
                df = tf.where(tf.logical_and(z,z_),l_,l) 
                
                fire_t_ = fire_t+df # Update firing time 
                
                return tf.concat([out[:-n_],fire_t_],0)
            else:
                return out
        return scan_func
    
    def _step_func(self, func, t, dt, y):
        k1 = func(y, t)
        half_step = t + dt / 2
        dt_cast = tf.cast(dt, y.dtype) # Failsafe

        k2 = func(y + dt_cast * k1 / 2, half_step)
        k3 = func(y + dt_cast * k2 / 2, half_step)
        k4 = func(y + dt_cast * k3, t + dt)
        return tf.add_n([k1, 2 * k2, 2 * k3, k4]) * (dt_cast / 6)
    

def odeint(func, y0, t, n_, F_b):
    t = tf.convert_to_tensor(t, preferred_dtype=tf.float64, name='t')
    y0 = tf.convert_to_tensor(y0, name='y0')
    tf_check_type(y0,t)
    with tf_check_increasing(t):
        return _Tf_Integrator(n_, F_b).integrate(func,y0,t)
