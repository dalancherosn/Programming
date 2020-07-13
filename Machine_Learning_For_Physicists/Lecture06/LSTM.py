# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:00:11 2020

@author: dalan
"""

##############################################################################
################# LIBRERÍAS INVOLUCRADAS #####################################
##############################################################################


# Import keras library. Also import some of the layers, so we do not need to
# write things like "layers.Dense", but can just write "Dense" instead
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Import the numpy library for matrix manipulations etc.
from numpy import *

# Set up the graphics by importing the matplotlib plotting library
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display


##############################################################################
#################### RECALL NET ##############################################
##############################################################################


def init_memory_net():
    global rnn, timesteps
    rnn = Sequential()
    # note: batch input_shape is (batchsize, timesteps, data_dim)
    rnn.add(LSTM(5, batch_input_shape = (None, timesteps, 3), 
                 return_sequences = True))
    rnn.add(LSTM(2, return_sequences = True))
    rnn.compile(loss = 'mean_squared_error', optimizer = 'adam', 
                metrics = ['accuracy'])
    
    
def produce_batch():
    global batchsize, timesteps
    
    observations=zeros([batchsize,timesteps,3])
    desired_output=zeros([batchsize,timesteps,2])
    
    tell_position=random.randint(int(timesteps/2),size=batchsize)
    ask_position=int(timesteps/2)+1+random.randint(int(timesteps/2)-1,size=batchsize)
    
    # mark input-slot 0 with 1 at the tell_position:
    observations[range(batchsize),tell_position,0]=1
    # write the value to be memorized into input-slot 1
    memorize_numbers=random.random(batchsize)
    observations[range(batchsize),tell_position,1]=memorize_numbers
    # mark input-slot 2 with 1 at the ask_position
    observations[range(batchsize),ask_position,2]=1
    
    desired_output[range(batchsize),ask_position,0]=memorize_numbers
    return(observations,desired_output)


timesteps=20


init_memory_net()

batchsize=1
test_observations,test_target=produce_batch()

batchsize=20
epochs=300

test_output=zeros([timesteps,epochs])

for k in range(epochs):
    input_observations,output_targets=produce_batch()
    rnn.train_on_batch(input_observations,output_targets)
    test_output[:,k]=rnn.predict_on_batch(test_observations)[0,:,0]
    print("epoch: ", k, " deviation: ", "{:1.3f}".format(sum((test_output[:,k]-test_target[0,:,0])**2)), end="      \r")

fig=plt.figure(figsize=(7,5))
plt.imshow(test_output,origin='lower',interpolation='nearest',aspect='auto')
plt.colorbar()
plt.show()


batchsize=30
test_observations,test_target=produce_batch()
test_output=zeros([batchsize,timesteps])

test_output[:,:]=rnn.predict_on_batch(test_observations)[:,:,0]

fig=plt.figure(figsize=(3,2))
plt.imshow(test_target[:,:,0],vmax=1.0,vmin=0.0,origin='lower',interpolation='nearest',aspect='auto')
plt.colorbar()
plt.show()

fig=plt.figure(figsize=(3,2))
plt.imshow(test_output,vmax=1.0,vmin=0.0,origin='lower',interpolation='nearest',aspect='auto')
plt.colorbar()
plt.show()

print("Deviation: ", sum((test_output-test_target[:,:,0])**2))



def produce_batch_tell_ask_twice():
    global batchsize, timesteps
    
    observations=zeros([batchsize,timesteps,3])
    desired_output=zeros([batchsize,timesteps,2])
    
    tell_position=random.randint(int(timesteps/2),size=batchsize)
    ask_position=int(timesteps/2)+1+random.randint(int(timesteps/4)-2,size=batchsize)
    ask_position2=ask_position+1+random.randint(int(timesteps/4)-2,size=batchsize)
    
    # mark input-slot 0 with 1 at the tell_position:
    observations[range(batchsize),tell_position,0]=1
    # write the value to be memorized into input-slot 1
    memorize_numbers=random.random(batchsize)
    observations[range(batchsize),tell_position,1]=memorize_numbers
    # mark input-slot 2 with 1 at the ask_position
    observations[range(batchsize),ask_position,2]=1
    observations[range(batchsize),ask_position2,2]=1
    
    desired_output[range(batchsize),ask_position,0]=memorize_numbers
    desired_output[range(batchsize),ask_position2,0]=memorize_numbers
    return(observations,desired_output)


batchsize=30
test_observations,test_target=produce_batch_tell_ask_twice()
test_output=zeros([batchsize,timesteps])

test_output[:,:]=rnn.predict_on_batch(test_observations)[:,:,0]

fig=plt.figure(figsize=(4,3))
plt.imshow(test_target[:,:,0],vmax=1.0,vmin=0.0,origin='lower',interpolation='nearest',aspect='auto')
plt.colorbar()
plt.show()

fig=plt.figure(figsize=(4,3))
plt.imshow(test_output,vmax=1.0,vmin=0.0,origin='lower',interpolation='nearest',aspect='auto')
plt.colorbar()
plt.show()

print("Deviation: ", sum((test_output-test_target[:,:,0])**2))



def init_memory_net_powerful():
    global rnn, batchsize, timesteps
    rnn = Sequential()
    # note: batch_input_shape is (batchsize,timesteps,data_dim)
    rnn.add(LSTM(20, batch_input_shape=(None, timesteps, 3), return_sequences=True))
    rnn.add(LSTM(2, return_sequences=True))
    rnn.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    
    
    
init_memory_net_powerful()

timesteps=20

batchsize=1
test_observations,test_target=produce_batch()

batchsize=20
epochs=300

test_output=zeros([timesteps,epochs])

for k in range(epochs):
    input_observations,output_targets=produce_batch()
    rnn.train_on_batch(input_observations,output_targets)
    test_output[:,k]=rnn.predict_on_batch(test_observations)[0,:,0]
    print("\r epoch: ", k, " deviation: ", sum((test_output[:,k]-test_target[0,:,0])**2), end="")


fig=plt.figure(figsize=(10,7))
plt.imshow(test_output,origin='lower',interpolation='nearest',aspect='auto')
plt.colorbar()
plt.show()

batchsize=30
test_observations,test_target=produce_batch()
test_output=zeros([batchsize,timesteps])

test_output[:,:]=rnn.predict_on_batch(test_observations)[:,:,0]

fig=plt.figure(figsize=(5,4))
plt.imshow(test_target[:,:,0],vmax=1.0,vmin=0.0,origin='lower',interpolation='nearest',aspect='auto')
plt.colorbar()
plt.show()

fig=plt.figure(figsize=(5,4))
plt.imshow(test_output,vmax=1.0,vmin=0.0,origin='lower',interpolation='nearest',aspect='auto')
plt.colorbar()
plt.show()

print("Deviation: ", sum((test_output-test_target[:,:,0])**2))


def init_count_net():
    global rnn, batchsize, timesteps
    global firstLSTMlayer
    rnn = Sequential()
    # note: batch_input_shape is (batchsize,timesteps,data_dim)
    firstLSTMlayer=LSTM(2, batch_input_shape=(None, timesteps, 2), return_sequences=True)
    rnn.add(firstLSTMlayer)
    rnn.add(LSTM(2, return_sequences=True))
    rnn.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    
    
def produce_batch_counting():
    global batchsize, timesteps
    
    observations=zeros([batchsize,timesteps,2])
    desired_output=zeros([batchsize,timesteps,2])
    
    tell_position=random.randint(int(timesteps/2),size=batchsize)
    count_position=random.randint(int(timesteps/2)-1,size=batchsize)
    expect_position=tell_position+count_position
    
    # mark input-slot 0 with 1 at the tell_position:
    observations[range(batchsize),tell_position,0]=1
    # write the counter value
    observations[range(batchsize),tell_position,1]=count_position
    
    desired_output[range(batchsize),expect_position,0]=1
    return(observations,desired_output)


timesteps=20


init_count_net()


batchsize=1
test_observations,test_target=produce_batch_counting()

batchsize=20
epochs=300

test_output=zeros([timesteps,epochs])

for k in range(epochs):
    input_observations,output_targets=produce_batch_counting()
    rnn.train_on_batch(input_observations,output_targets)
    test_output[:,k]=rnn.predict_on_batch(test_observations)[0,:,0]
    print("epoch: ", k, " deviation: ", "{:1.3f}".format(sum((test_output[:,k]-test_target[0,:,0])**2)), end="    \r")



fig=plt.figure(figsize=(7,5))
plt.imshow(test_output,origin='lower',interpolation='nearest',aspect='auto')
plt.colorbar()
plt.show()



from tensorflow.keras import Model

# get a function that represents the mapping from the 
# network inputs to the neuron output values of the first LSTM layer:
neuron_values = Model([rnn.inputs], [firstLSTMlayer.output])



batchsize=1
test_observations,test_target=produce_batch_counting()



print(test_observations)


the_values=neuron_values.predict_on_batch([test_observations])


shape(the_values)


plt.imshow(the_values[0,:,:],interpolation='nearest',origin='lower',aspect='auto')
plt.colorbar()
plt.show()


