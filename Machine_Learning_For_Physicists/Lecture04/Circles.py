# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:10:33 2020

@author: dalan
"""

# keras: Sequential is the neural-network class, Dense is
# the standard network layer

from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, UpSampling2D
from tensorflow.keras import optimizers # to choose more advanced optimizers like 'Adam'

import numpy as np

import matplotlib.pyplot as plt # for plotting
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display 

# for subplots within subplots:
from matplotlib import gridspec

# for nice inset colorbars: (approach changed from lecture 1 'Visualization' notebook)
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

# for updating display
# (very simple animation)
from IPython.display import clear_output
from time import sleep

# visualization routines:
    
def visualize_CNN_training(network,
                               image_generator, resolution,
                               steps = 100, batchsize = 10,
                               visualize_nsteps = 1, plot_img_pixels = 3,
                               plot_img_cols = 10,
                               plot_img_rows = 5,
                               show_intermediate_layers = True):
    """
    Visualize the training of a (2D) convolutional neural network autoencoder.
    

    Parameters
    ----------
    network : A net of tensorflow
        is the network that you defined using keras
    image_generator : Function
        is the name of a function that is called like image_generator(batchsize, x, y)
        and which has to return an array of shape [batchsize, M, M] that contains 
        randomly generated MxM images (e.g. randomly placed circles or whatever you want to consider).
        The MxM arrays x and y are already filled with coordinates between -1 and 1.
        
        An example that returns images of randomly placed circles:
            
        def my_generator(batchsize, x, y):
            R = np.random.uniform(size = batchsize)
            x0 = np.random.uniform(size = batchsize, low = -1, high = 1)
            y0 = np.random.uniform(size = batchsize, low = -1, high = +1)
            return( 1.0*((x[None, :, :] - x0[:, None, None])**2 + 
                         (y[None, :, :]- y0[None, :, :])**2 < R[:, None, None]**2) )
    resolution : Integer number
        DESCRIPTION. (called M below) is the image resolution in pixels.
    steps : Integer number 
        DESCRIPTION. Is the number of training steps. The default is 100.
    batchsize : Integer number
        DESCRIPTION. Is the number of samples per training step. The default is 10.
    visualize_nsteps > 1: Integer number
        DESCRIPTION. means skip some steps before visualizing again. The default is 1.
        (can speed up things)
    show_intermediate_layers = True : logical
        DESCRIPTION. Means show the intermediate activations.
        Otherwise show the weights!
        
    These are always shown in the upper left corner, as a tilted image,
    whose properties are determined by:
        plot_img_pixels: the resolution for each of the image tiles
        plot_img_cols: the number of columns of images
        plot_img_rows: the numbr of rows of images
    Images (activations or weights) that are larger will be cut off.
    If there are more images than fit, the rest will be left out.
    The lowest layer starts in the bottom left. For activations, for
    each layer one runs through all the channels, and then the images
    for the next layer will start. Likewise for weights.

    Returns
    -------
    None.

    """

    global y_target # allow access to target from outside
    
    M = resolution
        
    vals = np.linspace(-1,1,M)
    x,y = np.meshgrid(vals,vals)
    
    y_test=np.zeros([1,M,M,1])
    y_test[:,:,:,0]=image_generator(1,x,y)
    
    y_in=np.zeros([batchsize,M,M,1])

    costs=np.zeros(steps)
    extractor=get_layer_activation_extractor(network)
    
    for j in range(steps):
        # produce samples:
        y_in[:,:,:,0]=image_generator(batchsize,x,y)
        y_target=np.copy(y_in) # autoencoder wants to reproduce its input!
        
        # do one training step on this batch of samples:
        costs[j]=network.train_on_batch(y_in,y_target)
        
        # now visualize the updated network:
        if j%visualize_nsteps==0:
            clear_output(wait=True) # for animation
            if j>10:
                cost_max=np.average(costs[0:j])*1.5
            else:
                cost_max=costs[0]
            
            # nice layout (needs matplotlib v3)
            fig=plt.figure(constrained_layout=True,figsize=(8,4))
            gs=fig.add_gridspec(ncols=8,nrows=4)
            filter_plot=fig.add_subplot(gs[0:3,0:4])
            cost_plot=fig.add_subplot(gs[3,0:4])
            test_in_plot=fig.add_subplot(gs[0:2,4:6])
            test_out_plot=fig.add_subplot(gs[0:2,6:8])

            cost_plot.plot(costs)
            cost_plot.set_ylim([0,cost_max])
            
            # test the network on a fixed test image!
            y_test_out=network.predict_on_batch(y_test)
            test_in_plot.imshow(y_test[0,:,:,0],origin='lower')
            test_out_plot.imshow(y_test_out[0,:,:,0],origin='lower')
            test_in_plot.axis('off')
            test_out_plot.axis('off')
            
            if show_intermediate_layers:
                features=extractor(y_test)
                n1=0; n2=0
                max_n1=plot_img_rows
                max_n2=plot_img_cols
                pix=plot_img_pixels
                img=np.full([(pix+1)*max_n1,(pix+1)*max_n2],1.0)
                for feature in features:
                    for m in range(feature.shape[-1]):
                        w=feature[0,:,:,m]
                        ws=np.shape(w)
                        if n1<max_n1 and n2<max_n2:
                            W=np.zeros([pix,pix])
                            if ws[0]<pix:
                                W[0:ws[0],0:ws[0]]=w[:,:]
                            else:
                                W[:,:]=w[0:pix,0:pix]                            
                            img[n1*(pix+1):(n1+1)*(pix+1)-1,n2*(pix+1):(n2+1)*(pix+1)-1]=W
                            n2+=1
                            if n2>=max_n2:
                                n2=0
                                n1+=1                
            else: # rather, we want the weights! (filters)
                n1=0; n2=0
                max_n1=plot_img_rows
                max_n2=plot_img_cols
                pix=plot_img_pixels
                img=np.zeros([(pix+1)*max_n1,(pix+1)*max_n2])
                for ly in network.layers:
                    w=ly.get_weights()
                    if w!=[]:
                        w=w[0]
                        ws=np.shape(w)
                        for k1 in range(ws[2]):
                            for k2 in range(ws[3]):
                                if n1<max_n1 and n2<max_n2:
                                    W=np.zeros([pix,pix])
                                    if ws[0]<pix:
                                        W[0:ws[0],0:ws[0]]=w[:,:,k1,k2]
                                    else:
                                        W[:,:]=w[0:pix,0:pix,k1,k2]                            
                                    img[n1*(pix+1):(n1+1)*(pix+1)-1,n2*(pix+1):(n2+1)*(pix+1)-1]=W
                                    n2+=1
                                    if n2>=max_n2:
                                        n2=0
                                        n1+=1
            filter_plot.imshow(img,origin='lower')
            filter_plot.axis('off')
            plt.show()
    print("Final cost value (averaged over last 50 batches): ", np.average(costs[-50:]))


def print_layers(network, y_in):
    """
    Call this on some test images y_in, to get a print-out of
    the layer sizes. Shapes shown are (batchsize,pixels,pixels,channels).
    After a call to the visualization routine, y_target will contain
    the last set of training images, so you could feed those in here.
    """
    layer_features=get_layer_activations(network,y_in)
    for idx,feature in enumerate(layer_features):
        s=np.shape(feature)
        print("Layer "+str(idx)+": "+str(s[1]*s[2]*s[3])+" neurons / ", s)

def get_layer_activation_extractor(network):
    return(Model(inputs=network.inputs,
                            outputs=[layer.output for layer in network.layers]))

def get_layer_activations(network, y_in):
    """
    Call this on some test images y_in, to get the intermediate 
    layer neuron values. These are returned in a list, with one
    entry for each layer (the entries are arrays).
    """
    extractor=get_layer_activation_extractor(network)
    layer_features = extractor(y_in)
    return(layer_features)










