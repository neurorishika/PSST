#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/neurorishika/PSST/blob/master/Tutorial/Optional%20Material/Distributed%20TensorFlow/Distributed%20TensorFlow.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp; <a href="https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/neurorishika/PSST/master/Tutorial/Optional%20Material/Distributed%20TensorFlow/Distributed%20TensorFlow.ipynb" target="_parent"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle"/></a>

# ## Day 7: (Optional) Distributed Computing with TensorFlow
# 
# TensorFlow supports distributed computing, allowing portions of the graph to be computed on different processes, which may be on completely different servers! In addition, this can be used to distribute computation to servers with powerful GPUs, and have other computations done on servers with more memory, and so on. Unfortunately, the official documentation on Distributed TensorFlow rather jumps in at the deep end. For a slightly more gentle introduction we will run through some really basic examples with Jupyter.

# In[ ]:


import tensorflow as tf
## OR ##
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# Most times when we write a distributed code, we want each server to have access to a common set of variables. Say we want to share the variable var between two sessions (called sess1 and sess2) created on two different processes on different clusters.

# In[ ]:


var = tf.Variable(initial_value=0.0)

# Imagine this was run on server 1
sess1 = tf.Session()
sess1.run(tf.global_variables_initializer())

# Imagine this was run on server 1
sess2 = tf.Session()
sess2.run(tf.global_variables_initializer())


# Whenever a call to tf.Session() is made, it creates a completely seperate "execution engine". It is then connected to the session handle and the execution engine that stores variable values and runs operations. Lets try making changes on the variable var.

# In[ ]:


print("Value of var (Session 1):", sess1.run(var))
print("Value of var (Session 2):", sess2.run(var))

sess1.run(var.assign_add(1.0))
print("Increment var in Session 1")

print("Value of var (Session 1):", sess1.run(var))
print("Value of var (Session 2):", sess2.run(var))


# Thus, we can see, sessions in different processes are unlinked. Changing var in one session (on one execution engine) won't affect var in the other session. In order to share variables between processes, we need to link the different execution engines together. This is where we introduce Distributed TensorFlow.
# 
# ### Distributed TensorFlow
# 
# TensorFlow works a bit like a server-client model. The idea is that the users creates a whole bunch of worker nodes that will perform the heavy lifting. A session is then created on one of those worker nodes, and it will compute the graph, possibly distributing parts of it to other worker nodes on the cluster.
# 
# In order to do this, the main worker, needs to know about the other workers. This is done via the creation of a "ClusterSpec", which you need to pass to all workers. A ClusterSpec is built using a dictionary, where the key is a “job name”, and each job contains many workers.
# 
# The first step is to define what the cluster looks like. We start off with the simplest possible cluster: two worker nodes/servers, both on the same machine; one that will listen on port 2020, one on port 2021. And we create a job called "local" using these servers.

# In[ ]:


# Define the Worker Nodes (called Tasks)
tasks = ["localhost:2020", "localhost:2021"]
# Define the Cluster Jobs which is a dictionary of connect tasks
jobs = {"local":tasks}
# Initialize the Cluster using ClusterSpec
cluster = tf.train.ClusterSpec(jobs)


# We now launch the servers associated with the cluster jobs using the Server function.

# In[ ]:


# This server corresponds to the the first worker associated with the 'local' job.
s1 = tf.train.Server(cluster, job_name="local", task_index=0)
# This server corresponds to the the second worker associated with the 'local' job.
s2 = tf.train.Server(cluster, job_name="local", task_index=1)


# With the servers linked together in the same cluster, variables in any one of the server will be shared between all servers. By default, variables and operations get stored and executed on the first worker in the cluster. but to fix a variable or an operation to a specific worker, we can use tf.device().

# In[ ]:


# Place variable 'var' in the first server
with tf.device("/job:local/task:0"): 
  var = tf.Variable(initial_value=0.0, name='var')
sess1 = tf.Session(s1.target)
sess2 = tf.Session(s2.target)


# We can now try the same thing we did earlier to change the value of var.

# In[ ]:


# Initialize the variables
sess1.run(tf.global_variables_initializer())
sess2.run(tf.global_variables_initializer())

print("Value of var (Session 1):", sess1.run(var))
print("Value of var (Session 2):", sess2.run(var))

sess1.run(var.assign_add(1.0))
print("Increment var in Session 1")

print("Value of var (Session 1):", sess1.run(var))
print("Value of var (Session 2):", sess2.run(var))


# Voila! Now the value of var is changed for both sessions. An interesting thing to note would be that the second tf.global_variables_initializer() is redundant as there is only a single shared variable that gets initialized by the first call.

# In[ ]:


with tf.device("/job:local/task:0"):
    var1 = tf.Variable(0.0, name='var1')
with tf.device("/job:local/task:1"):
    var2 = tf.Variable(0.0, name='var2')
    
# (This will initialize both variables)
sess1.run(tf.global_variables_initializer())


# Here, whenever we use var1 it will always be run on the first task/worker node (localhost:2020) and for var2 it will always be run on the second task/worker node (localhost:2021).
# 
# ### Example
# 
# Lets try to take a simple Tensorflow Computation graph and split it across multiple processes.

# In[ ]:


#@markdown Restart runtime
exit()


# In[ ]:


import tensorflow as tf
## OR ##
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x = tf.constant(2)
y1 = x + 300
y2 = x - 66
y = y1 + y2

with tf.Session() as sess:
    result = sess.run(y)
    print(result)


# Now we will use Process() from the multiprocessing package to create workers on different processes and run the code.

# In[ ]:


#@markdown Restart runtime
exit()


# In[ ]:


from multiprocessing import Process

# Make a function that creates workers on "localhost:2020" 
# and "localhost:2021" given the worker number and joins

def create_server(worker_number):
    
    import tensorflow as tf
    ## OR ##
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    cluster = tf.train.ClusterSpec({"local": ["localhost:2020", "localhost:2021"]})
    worker = tf.train.Server(cluster, job_name="local", task_index=worker_number)
    print("Starting server #{}".format(worker_number))
    worker.start()
    worker.join()


# We then create and start the two processes using the function create_server and giving the worker number as the argument. 

# In[ ]:


# Create Process
p1 = Process(target=create_server,args=(0,))
p2 = Process(target=create_server,args=(1,))

# Start Process
p1.start()
p2.start()


# Finally we actually run the session.

# In[ ]:


import tensorflow as tf
## OR ##
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Initialize the Cluster we are using
cluster = tf.train.ClusterSpec({"local": ["localhost:2020", "localhost:2021"]})

x = tf.constant(2)

# define the device to use. FORMAT: /job:JOB_NAME/task:TASK_NUMBER

with tf.device("/job:local/task:0"):
    y2 = x - 66

with tf.device("/job:local/task:1"):
    y1 = x + 300
    y = y1 + y2

# Run session on one of the workers
with tf.Session("grpc://localhost:2020") as sess:
    result = sess.run(y)
    print(result)


# Thus we are now capable of running distributed code over TensorFlow. In an actual distributed scenario, we will be running the code defined in create_server() on different nodes of a cluster and run the last cell in the main worker node to actually perform the computation.
# 
# From this example it is really easy to now break the integrator into different sections and run them on different nodes to optimize performance by distributing some intensive computation to servers with powerful GPUs, and have other memory heavy computations done on servers with more memory, and so on. A device can be specified on a remote computer by modifying the device string. As an example “/job:local/task:0/gpu:0” will target the GPU on the local job.
# 
# Sources: 
# https://databricks.com/tensorflow/distributed-computing-with-tensorflow
# http://amid.fish/distributed-tensorflow-a-gentle-introduction
