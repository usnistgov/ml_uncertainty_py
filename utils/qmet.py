import scipy as sp
import scipy.spatial 
import scipy.stats
import matplotlib

#matplotlib.rcParams['text.usetex'] = True
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import copy
import math
from sklearn import decomposition as skdecomp

take_log = False


def fanndataread(filename):
    """Reads and normalizes a set of vector data in the file specified by filename
    
    The datafile is assumed to be in the format from the Fast Artificial Neural Network Library. This format is:
     * One line that specifies the number of samples, the number of features in a sample, and the number of classes
     * Two lines per sample
     ** The feature vector for that sample
     ** The class vector for that sample
    
    Returns: 
     * featuredata, an array of the feature data
     * classdata, an array of the class identifications
     * fileinfo, an array containing the number of samples, the number of features, and the number of classes
    
    :param filename: The file from which the neural network data are to be read
    :type filename: str    
    """
    print(filename)
    annfile = open(filename,'r') # Open the FANN-formatted data file
    #Read the FANN-formatted NMR file
    annfile.seek(0) #Make sure we're at the beginning of the file
    fileinfo = np.fromstring(annfile.readline(),sep=' ',dtype=int) #Read: Number of data elements, number of bins, number of class IDs
    featuredata = np.empty((fileinfo[0],fileinfo[1])) #Build the empty anndata array
    classdata = np.empty((fileinfo[0],fileinfo[2]))
    for i in range(0, int(fileinfo[0]) ):
        featureline = np.fromstring(annfile.readline(),sep=' ') #read the 1800-column ANN data line
        classline = np.fromstring(annfile.readline(),sep=' ') #this is the class data, which isn't used for a SOM
        featuredata[i] = featureline #Load the feature data and class data into the arrays
        classdata[i] = classline
    annfile.close
    return featuredata,classdata,fileinfo
def Kohonen_udistance(i,j,Klayer):
    """Calculates the Kohonen U-matrix for a rectangular Kohonen map
    
    :param i:
    :param j:
    :param Klayer:
    :type i: int
    :type j: int
    :type Klayer: Kohonen codebook vector layer of a PyMVPA self-organizing map
    """
    uvec = Klayer[i,j,:] #Get the Kohonen codebook vector at the current position
    dx = np.linalg.norm(Klayer[i+1,j,:]-uvec)
    dy = np.linalg.norm(Klayer[i,j+1,:]-uvec)
    dxy = np.linalg.norm(Klayer[i+1,j+1,:]-uvec)
    dyx = np.linalg.norm(Klayer[i,j+1,:]-Klayer[i+1,j,:])
    dz = (dxy + dyx)/2
    return dx,dy,dz
def spectrumplot(data,increment,vertsize,maxz,symmetric=False,xdata=np.array([0])):
    """Plots a set of one-dimensional spectra as a heat map. The data are split into several heat maps to facilitate visualization
    
    :param data: The array of spectral data to be plotted
    :param increment: the increment that each sub map includes
    :param vertsize: The size of the plot in inches
    :param maxz: The maximum value for the heat map
    :param symmetric: If False or absent, then the minimum value for the heat map is 0. Otherwise it is -1*maxz
    :param xdata: The x values corresponding to the spectral data, if any
    """
    data_xrange = data.shape[1]
    print(data_xrange)
    numplots = int(math.floor(data_xrange/float(increment)))
    mfig,axes = plt.subplots(numplots,1,figsize=(20,vertsize))
    vertextent = data.shape[0]
    if symmetric:
        minz = -maxz
        colormap="seismic"
    else:
        minz = 0
        colormap="Greys"
    
        
    for i,ax in enumerate(axes):
        if xdata.any():
            extent = [xdata[i*increment],xdata[(i+1)*increment - 1],0,vertextent]
        else:
            extent = [i*increment,(i+1)*increment - 1,0,vertextent]
        im = ax.imshow(data[:,i*increment:(i+1)*increment],
                       origin="lower",cmap=colormap,interpolation="none",
                       extent=extent,vmin=minz,vmax=maxz,aspect="auto")
    cax = mfig.add_axes([0.92,0.12,0.01,0.78])
    mfig.colorbar(im,cax=cax)
    return
def hellinger(x,y):
    """Computes the Hellinger distance between two sum-normalized vectors
    
    :param x: The first vector
    :param y: The second vector
    :type x: ndarray
    :type y: ndarray
    :returns: ``hellinger(x,y) = np.sqrt( 1 - np.dot(np.sqrt(x),np.sqrt(y)) )``
    """
    
    #hellinger_distance = sp.spatial.distance.euclidean(np.sqrt(x),np.sqrt(y))
    squared_hellinger_distance = 1 - np.dot(np.sqrt(x),np.sqrt(y))
    hellinger_distance = np.sqrt(squared_hellinger_distance)
    return hellinger_distance
def hellinger_hyp(x,y):
    """Computes the hyperbolic Hellinger distance metric between two vectors
    
    :param x: The first vector
    :param y: The second vector
    :type x: ndarray
    :type y: ndarray
    :returns: ``hellinger_hyp(x,y) = math.log( (1+hellinger(x,y))/(1 - hellinger(x,y)) )``
    """
    hellinger_distance = hellinger(x,y)
    try:
        hellinger_hyperbolic_distance = math.log( (1 + hellinger_distance)/(1 - hellinger_distance) )
    except:
        print(hellinger_distance)
    return hellinger_hyperbolic_distance
def jeffries(x,y):
    """Computes the Jeffries divergence between two vectors
    
    :param x: The first vector
    :param y: The second vector
    :type x: ndarray
    :type y: ndarray
    :returns:``jeffries(x,y) = (1/2) * ( entropy(x,y) + entropy(y,x) )``
    
    """
    entropy_xy = sp.stats.entropy(x,y)
    entropy_yx = sp.stats.entropy(y,x)
    jeffries_entropy = (entropy_xy+entropy_yx)/2.0
    return jeffries_entropy
def jensen(x,y):
    """Computes the Jensen-Shannon metric between two vectors
    
    :param x: The first vector
    :param y: The second vector
    :type x: ndarray
    :type y: ndarray
    :returns:``jensen(x,y) = entropy(x,m) + entropy(y,m)`` where ``m = (1/2) * (x + y)``
    
    """
    mean = (x+y)/2.0
    entropy_xm = sp.stats.entropy(x,mean)
    entropy_ym = sp.stats.entropy(y,mean)
    jensen_entropy = (entropy_xm + entropy_ym)/(2.0 * math.log(2))
    return jensen_entropy
def jensen_distance(x,y):
    """Computes the Jensen-Shannon distance between two vectors
    
    :param x: The first vector
    :param y: The second vector
    :type x: ndarray
    :type y: ndarray
    :returns:``jensen_distance(x,y) = sqrt(entropy(x,m) + entropy(y,m))``
    
    where ``m = (1/2) * (x + y)``
    
    """
    jensen_entropy = jensen(x,y)
    jensen_distance = np.sqrt(jensen_entropy)
    return jensen_distance
def jensen_hyp(x,y):
    """Computes the hyperbolic Jensen-Shannon metric between two vectors
    
    :param x: The first vector
    :param y: The second vector
    :type x: ndarray
    :type y: ndarray
    :returns: ``jensen_hyp(x,y) = math.log( (1+jensen_distance(x,y))/(1 - jensen_distance(x,y)) )``
    """
    jensen_entropy = jensen(x,y)
    jensen_distance = np.sqrt(jensen_entropy)
    jensen_hyperbolic_distance = math.log( (1+jensen_distance)/(1 - jensen_distance) )
    return jensen_hyperbolic_distance
def fix_spectrum(anndata):
    """Removes small values from an NMR spectrum and replaces them with an arbitrarily small value, in this case 1.0e-16
    
    :param anndata: The data to be corrected
    :returns: anndata_fixed, the corrected data
    """
    anndata_fixed = copy.deepcopy(anndata)
    eps = 1e-16
    anndata_fixed[anndata < eps] = eps
    return anndata_fixed
def distance_square(data,metric):
    distance_condensed = sp.spatial.distance.pdist(data,metric)
    distance_square = sp.spatial.distance.squareform(distance_condensed)
    return distance_square
def positional_array(numspecs,numsets):
    """Creates the positional array vector based on the number of laboratories in an interlab study and the number of samples measured by each laboratory
    
    :param numspecs: The number of samples contained within each data set, or the number of objects that have been measured
    :param numsets: The number of laboratories in the intercomparison
    :returns: Pos, an array identifying the first and last elements of the data set from each laboratory
    """
    Pos = np.zeros((numsets,2),dtype=int)
    for i in range(numsets):
        Pos[i] = [i*numspecs,(i+1)*numspecs]
    return Pos
def data_reshape(data_bylab,positional_array):
    numsets = positional_array.shape[0]
    numspecs = positional_array[0,1]
    data_byspec = np.zeros_like(data_bylab)
    for i,ii in enumerate(range(numspecs)):
        for k,pos_small in enumerate(positional_array):
            data_byspec[k+i*numsets,:]=data_bylab[pos_small[0]+i,:]
    return data_byspec
def mahalanobis_covariance(data,positional_array,threshold=10e-2):
    """Calculates the pooled covariance matrix among spectra for use in determining the Mahalanobis distance.
    
    :param data: The data for which a Mahalanobis distance will be calculated
    :param positional_array: The positional array that defines how the data are aligned
    :param threshold: The threshold for singular values to retain in the inverse covariance matrix
    :type threshold: float
    :returns: mahalanobis_dict
    """
    numsets = positional_array.shape[0]
    numspecs = positional_array[0,1]
    
    covariance = None
    
    for spec_num in range(numspecs):
        spec_array = positional_array[:,0]+spec_num
        data_array = data[spec_array,:]
        data_centered = data_array - data_array.mean(axis=0)
        cov_this = np.cov(data_centered.T)
        if covariance is None:
            covariance = np.zeros_like(cov_this)
        covariance += (numsets - 1) * cov_this
    covariance = covariance / ((numsets * numspecs) - numspecs)
    VI=np.linalg.pinv(covariance,rcond=threshold)
    significant_components=np.arange(data.shape[1])
    mahalanobis_dict = dict(VI=VI,significant_components=significant_components)
    return mahalanobis_dict
def distances(data,metric,positional_array,mahalanobis_dict=None):
    """Calculates the interlaboratory distances for each spectrum and interspectral distances for each laboratory
    Must define the inverse covariance matrix if metric='mahalanobis'
    
    :param data: The array of spectral data whos distances must be calculated
    :param metric: The distance metric to be used for the calculation. Will be passed to scipy.spatial.distance.pdist()
    :param positional_array: The array defining the structure of the data matrix
    :param mahalanobis_dict: A dictionary containing the inverse covariance matrix among the spectra and the list of significant comonents used to calculate that matrix. Can be calculated using qmet.mahalanobis_covariance
    """
    
    VI=None
    significant_components=data.shape[0]
    
    if metric == 'mahalanobis':
        if not(mahalanobis_dict):
            raise ValueError('Must define the inverse covariance matrix if metric is mahalanobis')
        VI=mahalanobis_dict['VI']
        significant_components=mahalanobis_dict['significant_components']
    
    
    numsets = positional_array.shape[0]
    numspecs = positional_array[0,1]
    
    total_sets = numsets*numspecs#positional_array[-1,-1]
    
    distance_bylab = np.zeros((total_sets,numspecs))
    distance_byspec = np.zeros((total_sets,numsets))
    
    for spec_num in range(numspecs):
        spec_array = positional_array[:,0]+spec_num
        #print spec_array
        data_array = data[spec_array,:]
        if metric == 'mahalanobis':
            data_array = data_array[:,significant_components]
        #print data_array.shape
        distance_array = sp.spatial.distance.pdist(data_array,metric=metric,VI=VI)
        distance_byspec[spec_num*numsets:(spec_num+1)*numsets,:] = sp.spatial.distance.squareform(distance_array)
        #plt.imshow( sp.spatial.distance.squareform(distance_array), origin='lower',cmap='Greys',interpolation="none")
    for lab_num in range(numsets):
        spec_array = range(positional_array[lab_num,0],positional_array[lab_num,1])
        #print spec_array
        data_array = data[spec_array,:]
        if metric == 'mahalanobis':
            data_array = data_array[:,significant_components]
        #print data_array.shape
        distance_array = sp.spatial.distance.pdist(data_array,metric=metric,VI=VI)
        distance_bylab[lab_num*numspecs:(lab_num+1)*numspecs,:] = sp.spatial.distance.squareform(distance_array)
    #raise ValueError
    return distance_bylab,distance_byspec
def distance_measure_fig(xdata,data,
                         positional_array,Spectrum_names,sets_to_plot,
                         distance_matrix_list,distance_labels,lab_labels,cmap='Greys',linecolor='k'):
    """Creates the plot of the distance measure figure, which will plot all of the samples corresponding to a given sample label and the interspectral distances
    
    :param xdata: The x-data labels corresponding to the spectral data
    :param ydata: The spectral data
    :param positional_array: The positional array describing how to find the spectral data in ydata that correspond to particular sample labels
    :param Spectrum_names: The names attached to the sample labels. This will be written on each row
    :param sets_to_plot: An iterable of integers describing which sample labels will be plotted
    :param distance_matrix_list: The list of distance matrices for which heat maps will be plotted
    :param distance_labels: The names of the distance metrics, which will be plotted in the header
    :param lab_labels: The names of the laboratories or data sets in the study
    :key cmap: The color map that will be used for the distance heat maps
    :key linecolor: The line color that will be used for the spectral data
    :returns: sfig, the distance measure figure matplotlib object.
    """
    numdist = len( distance_matrix_list )

    
    numsets = positional_array.shape[0]
    numspecs = positional_array[0,1]
    
    numrows = len(sets_to_plot)
    #print numrows
    
    #numspecs = 1
    
    spectrum_width = 2
    distance_width = 1.25
    
    height = 1.5
    label_buffer = 0.5
    title_buffer = 0.5
    numrows_eff = numrows + label_buffer + title_buffer
    width = 1.2 * height * (numdist*distance_width + spectrum_width + label_buffer)
    #total_height = height * (numrows_eff + 0.2)
    total_height = height * (numrows_eff)
    #sfig = plt.figure(figsize=(width,total_height))
    
    #sfig = plt.figure(figsize=(width,height * numrows + 1.25))
       
    
    #
    widths = [spectrum_width]
    total_width = spectrum_width
    numcols = numdist + 1
    #numcols = 2*numdist + 1
    for i in range(numdist):
        widths += [distance_width]#,0.2] # Add the appropriate number of columns to the width list
        total_width += distance_width#+0.2
    
    gs_key = dict(width_ratios=widths,hspace=0.075,wspace=0.00)
    sfig,sfig_axes = plt.subplots(numrows,numcols,figsize=(width,total_height),gridspec_kw=gs_key)
    
    if numrows < 2: sfig_axes = sfig_axes[np.newaxis,:]
    
    grids = gs.GridSpec(numrows,numcols,width_ratios = widths)
    
    #axlist = np.array([plt.subplot(grids[0])])
    #for i in range(numdist):
    #    ax = np.array([plt.subplot(grids[i+1])])
    #    axlist = np.concatenate((axlist,ax))
    
    ticks_master = np.array([0,0.2,1,2,4,8,16])
    #ticks_master = np.array([0,0.5,1.0])
    
    #for spec_num in range(numspecs):
    for plot_num,spec_num in enumerate(sets_to_plot):
        
        #ax0 = plt.subplot2grid( (numrows_eff,numdist + spectrum_width), (plot_num,0),colspan = spectrum_width)
        
        #ax0 = plt.subplot(grids[numcols*plot_num])
        ax0 = sfig_axes[plot_num,0]
        
        #ax0 = axlist[0]
        spec_array = positional_array[:,0]+spec_num
        pl0 = ax0.plot(xdata,np.transpose(data[spec_array,:])*35+np.array(range(numsets))-0.5/float(1)
                       ,linecolor)
        ax0.set_ylim([-0.5,numsets-0.5])
        ax0.invert_xaxis()
        ax0.text(9.5,numsets-1,Spectrum_names[spec_num][:3],ha='center', va='center',color='k',size=9,
                 bbox=dict(facecolor='w',alpha=1,lw=0,pad=0))
        ax0.set_yticks( np.arange(numsets) )
        ax0.set_yticklabels(lab_labels,size=7)
        
        ax0.set_xticklabels([])
        
        log_base = 1.01

        for dist_num in range(numdist):
            #ax = axlist[dist_num+1]
            #ax = plt.subplot(grids[numcols*plot_num + dist_num + 1])
            col_num = dist_num + 1
            #col_num = 2*dist_num+1
            ax = sfig_axes[plot_num,col_num]
            #ax = plt.subplot2grid( (numrows_eff,numdist + spectrum_width), (plot_num,spectrum_width + dist_num))
            
            distance_matrix = distance_matrix_list[dist_num]
            label = distance_labels[dist_num]
            #vmax = vmax_list[dist_num]
            
            datarray = distance_matrix[spec_num*numsets:(spec_num+1)*numsets,:]
            data_max = distance_matrix.max()
            log_max = math.log(data_max,10)
            decimals_max = int(-1*math.floor(log_max))+1
            significance_max = 10 ** decimals_max
            distance_max = math.ceil(data_max * significance_max)/significance_max
            
            data_min = distance_matrix[distance_matrix > 0].min()
            log_min = math.log(data_min,10)
            decimals_min = int(-1*math.floor(log_min))
            significance_min = 10 ** decimals_min
            distance_min = math.floor(data_min * significance_min)/significance_min
            
            #distance_max = math.ceil(math.pow(log_base,math.ceil(math.log(distance_matrix.max()*1000,log_base))))/1000
            #distance_min = math.floor(math.pow(log_base,math.floor(math.log(distance_matrix[distance_matrix > 0].min()*100,log_base))))/100
            
            #pl = ax.imshow(datarray,origin='lower',cmap='Greys',interpolation="none",vmin=0,vmax=vmax)
            pl = ax.imshow(datarray,origin='lower',cmap=cmap,interpolation="none",aspect='equal',vmin=distance_min,vmax=distance_max)
            if plot_num == 0:
                ax.set_title(label.replace(' ','\n'),size=10)
            ax.set_xticks( np.arange(numsets) )
            ax.set_yticks( np.arange(numsets) )
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if (plot_num + 1) == numrows:
                ax.set_xticklabels(lab_labels,rotation='vertical',size=7)
            ax.get_yaxis().set_tick_params(direction='out')
            ax.get_xaxis().set_tick_params(direction='out')
            ticks = ticks_master[ticks_master <= distance_max]
            #print ticks
            if (distance_max > 1.1 * ticks[-1]):
                ticks = np.concatenate((ticks,np.array([distance_max])))#[math.floor(distance_max)])))
            #ticks = np.array([distance_min,distance_max])
            #cax = sfig_axes[plot_num,col_num+1]
            cb = sfig.colorbar(pl,ax=ax,ticks=ticks,fraction=0.2)
            cb.ax.tick_params(labelsize=7)
    ax0_textlabel = r'Frequency shift / 10$^{-6}$ (ppm)'
    ax0.set_xlabel(ax0_textlabel,size=15)
    xticks = ax0.get_xticks()# * -1.0
    xticks_string = ["%2.0f" % xtick for xtick in xticks]
    ax0.set_xticklabels(xticks_string)
    
    y_axis_label_position = (label_buffer + numrows/2.0) / (numrows_eff)
    
    pairwise_distance_label = r'Pairwise Distance, $D_{ij,k}$'
    if numrows < 2: pairwise_distance_label = u'Pairwise\nDistance, $D_{ij,k}$'
    
    sfig.text(0.0,y_axis_label_position,'Data set ID',rotation='vertical',ha='center',va='center',size=15)
    sfig.text(1.0,y_axis_label_position,pairwise_distance_label,rotation='vertical',ha='center',va='center',size=15)
    
    buffer_in_left = 0.6
    buffer_in_right = 1 - buffer_in_left
    
    lab_title_x_position = (spectrum_width + buffer_in_left*label_buffer + (numdist*distance_width)/2.0) / (spectrum_width + numdist*distance_width + label_buffer)
    lab_title_y_position = 0#label_buffer/(2.0*numrows_eff)
    #print(lab_title_y_position)
    #print(lab_title_y_position*2)
    spectrum_title_x_position = (2.0) / (spectrum_width + numdist)
    #print lab_title_y_position
    #lab title = 
    #sfig.text(lab_title_x_position,lab_title_y_position,'Data set ID',ha='center',va='bottom',size=15)
    sfig.text(lab_title_x_position,lab_title_y_position,'Data set ID',ha='center',va='bottom',size=15)
    #sfig.text(lab_title_x_position,0.0,'Data set ID',ha='center',va='center',size=15)
    #sfig.text(lab_title_x_position,0.0,'Data set ID',ha='center',va='center',size=15)
    #sfig.text(spectrum_title_x_position,0.0,ax0_textlabel,ha='center',va='center',size=15)
    
    #for dist_num in range(numdist):
    #    metric_x_position = 
    

    
    #sfig.savefig('distances_fig.png',bbox_inches='tight',facecolor=[0,0,0,0])
    #sfig.tight_layout()
    
    #Adjust the subplot margins manually instead of using tight_layout() to do it, because the results of tight_layout() are unpredictable and ofthen don't play nicely with 
    
    left_buffer = buffer_in_left*label_buffer/(spectrum_width + numdist + label_buffer)
    right_buffer = 1 - buffer_in_right*label_buffer/(spectrum_width + numdist + label_buffer)
    bottom_buffer = label_buffer/(numrows_eff)
    top_buffer = 1 - title_buffer/(numrows_eff)
    sfig.subplots_adjust(left=left_buffer,
                         right=right_buffer,
                         wspace=0.1,
                         bottom=bottom_buffer,
                         top=top_buffer)
    
    #This section is for plotting various markers on the plot in order to get the proportions right. It should not be used for production figures
    
    #ypos_fig_boundaries = -1/4.0 * 1.0/total_height
    #ypos_fig_bdy_inches = -1/2.0 * 1.0/total_height
    
    #sfig.text(left_buffer,ypos_fig_boundaries,'0.0',ha='center',va='center',size=10)
    #sfig.text(left_buffer,ypos_fig_bdy_inches,'{:4.2f}'.format(left_buffer*width),ha='center',va='center',size=10)
    #this_length = 0
    #for this_loc in widths:
    #    this_length += this_loc/total_width
    #    xpos = (this_length * right_buffer) + ((1-this_length) * left_buffer)
    #    sfig.text(xpos,ypos_fig_boundaries,'{:4.2f}'.format(this_length),ha='center',va='center',size=10)
    #    sfig.text(xpos,ypos_fig_bdy_inches,'{:4.2f}'.format(xpos*width),ha='center',va='center',size=10)
    #
    #for point in np.linspace(0,1,11):
    #    x_point_pos = (point * right_buffer) + ((1-point) * left_buffer)
    #    sfig.text(x_point_pos,0,str(point),ha='center',va='center',size=10)
    #    #sfig.text(point,0,str(point),ha='center',va='center',size=10)
    #    y_label_pos = (point*top_buffer + (1-point)*bottom_buffer)
    #    #sfig.text(0,y_label_pos,str(point),ha='center',va='center',size=10)
    return sfig
def lognormal_integral(x,shape,scale=0):
    erfarg = (np.log(x) - scale)/(math.sqrt(2)*shape)
    lognorm_int = 1.0/2*(1+sp.special.erf(erfarg))
    return lognorm_int
def lognormal_z(probability):
    erfarg = 2*probability - 1
    z = np.exp(math.sqrt(2.0) * (sp.special.erfinv(erfarg)))
    return z
def get_zscores(distance_matrix,positional_array,distribution):
    """Calculates the Z scores among distances based on a lognormal distribution
    
    :param distance_matrix: The distance matrix for which the Z scores will be calculated
    :param positional_array: The array defining the structure of the data matrix
    :param distribution: Which distribution will be assumed when assigning Z scores to each measurement of a sample.
    
    :returns: Z scores, average interspectral distance, and the parameters of the 
    """
    numsets = positional_array.shape[0]
    numspecs = positional_array[0,1]
    
    lognorm_int_big = np.zeros_like(distance_matrix[:,0])
    distances_big = np.zeros_like(distance_matrix[:,0])
    zscores_big = np.zeros_like(distance_matrix[:,0])
    pdf_params = np.zeros_like(distance_matrix[:,0:3])
    
    for i in range(numspecs):
        #Get the data array for this spectrun
        datarray = distance_matrix[i*numsets:(i+1)*numsets,:]
        distance_sum = (datarray.sum(axis=1)+datarray.sum(axis=0))/2 #Take the sum across columns and rows
        distance_sum = distance_sum / (numsets - 1)
        distances_big[i*numsets:(i+1)*numsets] = distance_sum 
        
        #Fit to a lognormal distribution
        ln_dist = np.log(distance_sum)
        dist_mean = ln_dist.mean()
        dist_scale = math.exp(dist_mean)
        dist_std = ln_dist.std()
        
        #dist_mean,loc,dist_std = distribution.fit(distance_sum,floc=0)
        params = distribution.fit(distance_sum,floc=0)
        dist_scale = math.exp(params[0])
        
        pdf_params[i,:] = np.array([dist_mean,dist_scale,dist_std])
        
        
        lognorm_int = lognormal_integral(distance_sum,dist_std,dist_mean)
        lognorm_z = lognormal_z(lognorm_int)
        
        rv  = distribution(*params)
        std = distribution(1)
        lognorm_int = rv.cdf(distance_sum)
        lognorm_z = std.ppf(lognorm_int)
        
        lognorm_int_big[i*numsets:(i+1)*numsets] = lognorm_int
        zscores_big[i*numsets:(i+1)*numsets] = lognorm_z
               
    
    return zscores_big,distances_big,pdf_params
def plot_zscores(distances,zscores,pdf_params,positional_array,Spectrum_names,Labels,num_bins=20,plot_range=None,numcols=2,metric=None):
    """Plots the sample-level average diameter distances and the associated Z scores
    
    :param distances: The sample-level average diameter distances
    :param zscores: The sample-level Z scores
    :param pdf_params: The parameters for the distribution that the Z scores are drawn from
    :param positional_array: The array defining the structure of the data matrix
    :param Spectrum_names: The list of sample names
    :param Labels: The list of laboratory identifiers 
    :param plot_range: Which samples to plot
    :param num_cols: The number of columns in the plot
    :param metric: The metric used to calculate the interspectral distances
    :type plot_range: list of ints
    """
    numsets = positional_array.shape[0]
    numspecs = positional_array[0,1]
    
    #print numsets,numspecs
    
    if plot_range is None: plot_range = range(numspecs)
    numplots = len(plot_range)
    
    #numcols = 2
    numrows = int(math.floor((numplots + numcols - 1)/ numcols))
    
    bottomplot = numrows - 1
    
    width = 5
    jefffig,jeffax_array = plt.subplots(numrows,numcols,figsize=(numcols*width,numrows+1),sharex=True,sharey=True)
    jeffax = jeffax_array.flatten('F')
    
    jefffig.text(0.00,0.5,r"Average Diameter Distance, $\^{D}_{i,k}$",rotation='vertical',ha='center',va='center',size=15)
    #jefffig.text(0.5,-0.9/numrows,'Data set ID',ha='center',va='top',size=15)
    jefffig.text(0.5,0.00,'Data set ID',ha='center',va='top',size=15)
    
    #print 0.9/numrows
    
    log_base = 1.01
    
    #distance_max = 2 ** math.ceil(math.log(distances.max(),2))
    distance_max = math.ceil(
        math.pow(
            log_base,math.ceil(
                math.log(
                    distances.max()*100,log_base
                )
            )))/100
    #distance_max = math.ceil(distances.max()/10.0)*10.0
    
    #distance_min = 2 ** math.floor(math.log(distances.min(),2))
    distance_min = math.floor(
        math.pow(
            log_base,math.floor(
                math.log(
                    distances.min(),log_base
                )
            )))
    data_max = distances.max()
    log_max = math.log(data_max,10)
    decimals_max = int(-1*math.floor(log_max))+1
    significance_max = 10 ** decimals_max
    distance_max = math.ceil(data_max * significance_max)/significance_max
    
    data_min = distances.min()
    log_min = math.log(data_min,10)
    decimals_min = int(-1*math.floor(log_min))
    significance_min = 10 ** decimals_min
    distance_min = math.floor(data_min * significance_min)/significance_min
    
    jeff_locator = plt.MaxNLocator(nbins=4,prune='both')
    
    width = 0.4
    for plot_num,spec_num in enumerate(plot_range):
        distances_this = distances[spec_num*numsets:(spec_num+1)*numsets]
        
        dist_mean = pdf_params[spec_num,0]
        dist_scale = pdf_params[spec_num,1]
        dist_std = pdf_params[spec_num,2]
        
        zscores_this = zscores[spec_num*numsets:(spec_num+1)*numsets]
        
        bar_colors = ['w'] * numsets
        bar_edge = ['k'] * numsets
        text_colors = ['k'] * numsets
        box_trans = [0.7] * numsets
        z_mask = zscores_this > 5
        for int_num,z_out in enumerate(z_mask):
            if z_out:
                bar_colors[int_num] = 'r'
                bar_edge[int_num] = 'r'
                #text_colors[int_num] = 'r'
                box_trans[int_num] = 0
        #jeffax[plot_num].bar(np.arange(numsets)-width/2,distances_this,2*width,alpha=1,color=bar_colors)
        jeffax[plot_num].bar(np.arange(numsets),distances_this,2*width,alpha=1,color=bar_colors,edgecolor=bar_edge)
        
        jeffax[plot_num].set_ylim(distance_min,distance_max)
        jeffax[plot_num].yaxis.set_major_locator(jeff_locator)
        jeff_yticks = jeffax[1].yaxis.get_majorticklocs()
        
        lab_label_pos = 0.95*distance_max + 0.05*distance_min
        
        sample_label_pos = 0.05*distance_max + 0.95*distance_min
        #print jeff_yticks[-1],jeff_yticks[0]
        #print distance_min,sample_label_pos
        if plot_num == 0:
            if metric is not None:
                jeffax[plot_num].set_title(metric)
        for int_num,zscore in enumerate(zscores_this):
            if(z_mask[int_num]): 
                label_pos = sample_label_pos
            else:
                label_pos = sample_label_pos + (distances_this[int_num] - distance_min)
            jeffax[plot_num].text(int_num,label_pos,
                           str(zscore)[0:4],
                           ha='center', va='bottom',
                           color=text_colors[int_num],size=9,
                           bbox=dict(facecolor='w',alpha=box_trans[int_num],lw=0)
                          )
            #if zscore < 5:
            #    jeffax[i].text(int_num,sample_label_pos,str(zscore)[0:4],ha='center', va='bottom',color="k",size=9)
            #else:
            #    jeffax[i].text(int_num,sample_label_pos,str(zscore)[0:4],ha='center', va='bottom',color="r",size=9)
        jeffax[plot_num].set_xlim(-0.5,numsets-0.5)
    for i in range(numcols):
        jeffax[bottomplot+i*numrows].set_xlim(-0.5,numsets-0.5)
        #jeffax[bottomplot+i*numrows].set_ylim(distance_min,distance_max)
        jeffax[bottomplot+i*numrows].set_xticks(np.arange(numsets))
        jeffax[bottomplot+i*numrows].yaxis.set_major_locator(jeff_locator)
        jeffax[bottomplot+i*numrows].set_xticklabels(Labels,rotation='vertical')
    lab_label_jeff_y_pos =  0.1*distance_min + 0.9*distance_max
    for plot_num,spec_num in enumerate(plot_range):
        jeffax[plot_num].text(0,lab_label_jeff_y_pos,Spectrum_names[spec_num],ha='left', va='top')
    jefffig.tight_layout()
    return jefffig
def plot_histograms(distances,zscores,pdf_params,positional_array,Spectrum_names,Labels,distribution,num_bins=20,plot_range=None,numcols=2):
    """Plots a histogram of the sample-level average diameter distances and the corresponding disribution function
    
    :param distances: The sample-level average diameter distances
    :param zscores: The sample-level Z scores
    :param pdf_params: The parameters for the distribution that the Z scores are drawn from
    :param positional_array: The array defining the structure of the data matrix
    :param Spectrum_names: The list of sample names
    :param Labels: The list of laboratory identifiers
    :param distribution: The distribution assumed when assigning Z scores to each measurement of a sample.
    :param num_bins: The number of bins to use in the histogram
    :param plot_range: Which samples to plot
    :param num_cols: The number of columns in the plot
    :param metric: The metric used to calculate the interspectral distances
    :type plot_range: list of ints
    """
    numsets = positional_array.shape[0]
    numspecs = positional_array[0,1]
    
    if plot_range is None: plot_range = range(numspecs)
    numplots = len(plot_range)
    
    #numcols = 2
    numrows = int(math.floor((numplots + numcols - 1)/ numcols))
    
    bottomplot = numrows - 1
    
    width = 5
    
    pdffig,pdfax_array = plt.subplots(numrows,numcols,figsize=(numcols*width,numrows),sharex=True,sharey=True)
    pdfax = pdfax_array.flatten('F')
    
    pdffig.text(0.00,0.5,"Probability Density",rotation='vertical',ha='center',va='center',size=15)
    #pdffig.text(0.5,0.01*(numrows-1),'Average Diameter Distance',ha='center',va='center',size=15)
    pdffig.text(0.5,0.0,r'Average Diameter Distance, $\^{D}_{i,k}$',ha='center',va='center',size=15)
    
    #print 0.01*(numrows-1)
    
    log_base = 1.01
    
    #distance_max = 2 ** math.ceil(math.log(distances.max(),2))
    distance_max = math.ceil(
        math.pow(
            log_base,math.ceil(
                math.log(
                    distances.max()*100,log_base
                )
            )))/100
    #distance_max = math.ceil(distances.max()/10.0)*10.0
    
    #distance_min = 2 ** math.floor(math.log(distances.min(),2))
    distance_min = math.floor(
        math.pow(
            log_base,math.floor(
                math.log(
                    distances.min(),log_base
                )
            )))
    data_max = distances.max()
    log_max = math.log(data_max,10)
    decimals_max = int(-1*math.floor(log_max))+1
    significance_max = 10 ** decimals_max
    distance_max = math.ceil(data_max * significance_max)/significance_max
    
    data_min = distances.min()
    log_min = math.log(data_min,10)
    decimals_min = int(-1*math.floor(log_min))
    significance_min = 10 ** decimals_min
    distance_min = math.floor(data_min * significance_min)/significance_min
    
    #print distance_min,distance_max
    #print significance_min,significance_max
    
    max_bin = distance_max
    min_bin = distance_min
    binsize = (max_bin - min_bin)/float(num_bins)
    
    pdf_locator = plt.MaxNLocator(nbins=4,prune='both')
    
    width = 0.4
    for plot_num,spec_num in enumerate(plot_range):
        distances_this = distances[spec_num*numsets:(spec_num+1)*numsets]
        
        dist_mean = pdf_params[spec_num,0]
        dist_scale = pdf_params[spec_num,1]
        dist_std = pdf_params[spec_num,2]
        params = distribution.fit(distances_this,floc=0)
        
        hist_norm,bins = np.histogram(distances_this,range=(min_bin,max_bin), bins=num_bins,density=True)
        #lognorm_dist = sp.stats.lognorm.pdf(bins,dist_std,scale=dist_scale)
        lognorm_dist = distribution.pdf(bins,*params)

        pdfax[plot_num].plot(bins,lognorm_dist,'k')
        pdfax[plot_num].bar(bins[:-1]-0.5*binsize,hist_norm,binsize,color='r')
    for i in range(numcols):
        pdfax[bottomplot+i*numrows].set_xlim(distance_min,distance_max)
        pdfax[bottomplot+i*numrows].yaxis.set_major_locator(pdf_locator)
    (ymin,ymax) = pdfax[bottomplot].get_ylim()
    lab_label_pdf_y_pos = 0.90*ymax + 0.10*ymin
    lab_label_pdf_x_pos = 0.95*distance_min + 0.05*distance_max
    for plot_num,spec_num in enumerate(plot_range):
        pdfax[plot_num].text(lab_label_pdf_x_pos,lab_label_pdf_y_pos,Spectrum_names[spec_num],ha='left', va='top')
    pdffig.tight_layout()
    return pdffig
def zscore_analysis(distance_matrix,positional_array,Spectrum_names,Labels,num_bins):
#    """Plots the Z scores for interlaboratory spectral distances, as well as a histogram 
#    and the corresponding probability density function
#    """
    
    zscores,distances,pdf_params = get_zscores(distance_matrix,positional_array)
    
    numsets = positional_array.shape[0]
    numspecs = positional_array[0,1]
    
    numcols = 2
    numrows = int(math.floor((numspecs + numcols - 1)/ numcols))
    
    bottomplot = numrows - 1
    
    jefffig,jeffax_array = plt.subplots(numrows,numcols,figsize=(10,5),sharex=True,sharey=True)
    pdffig,pdfax_array = plt.subplots(numrows,numcols,figsize=(10,5),sharex=True,sharey=True)
    
    jeffax = jeffax_array.flatten('F')
    pdfax = pdfax_array.flatten('F')
    #jeffbig = jeffig.add_subplot(111)
    
    jefffig.text(0.07,0.5,"Average Diameter Distance",rotation='vertical',ha='center',va='center',size=15)
    jefffig.text(0.5,-0.15,'Data set ID',ha='center',va='center',size=15)
    
    #jefffig.text(0,0.5,"Distance Score",rotation='vertical',ha='center',va='center',size=15)
    #jefffig.text(0.5,0,'Lab ID',ha='center',va='center',size=15)
    
    pdffig.text(0.07,0.5,"Probability Density",rotation='vertical',ha='center',va='center',size=15)
    pdffig.text(0.5,0.05,'Average Diameter Distance',ha='center',va='center',size=15)
    
    log_base = 1.01
    
    #distance_max = 2 ** math.ceil(math.log(distances.max(),2))
    distance_max = math.ceil(
        math.pow(
            log_base,math.ceil(
                math.log(
                    distances.max()*10,log_base
                )
            )))/10
    #distance_max = math.ceil(distances.max()/10.0)*10.0
    
    #distance_min = 2 ** math.floor(math.log(distances.min(),2))
    distance_min = math.floor(
        math.pow(
            log_base,math.floor(
                math.log(
                    distances.min(),log_base
                )
            )))
    data_max = distances.max()
    log_max = math.log(data_max)
    decimals_max = int(-1*math.floor(log_max))+1
    significance_max = 10 ** decimals_max
    distance_max = math.ceil(data_max * significance_max)/significance_max
    
    data_min = distances.min()
    log_min = math.log(data_min)
    decimals_min = int(-1*math.floor(log_min))+1
    significance_min = 10 ** decimals_min
    distance_min = math.floor(data_min * significance_min)/significance_min
    #distance_ticks = [math.floor(distance_max/3.0),math.floor(distance_max*2/3.0),distance_max-1]
    
    max_bin = distance_max
    min_bin = distance_min
    binsize = (max_bin - min_bin)/float(num_bins)
    
    jeff_locator = plt.MaxNLocator(nbins=4,prune='both')
    pdf_locator = plt.MaxNLocator(nbins=4,prune='both')
    
    width = 0.4
    for i in range(numspecs):
        
        distances_this = distances[i*numsets:(i+1)*numsets]
        #print np.arange(numsets)-width
        #print distances_this
        
        
        
        dist_mean = pdf_params[i,0]
        dist_scale = pdf_params[i,1]
        dist_std = pdf_params[i,2]
        
        zscores_this = zscores[i*numsets:(i+1)*numsets]
        
        bar_colors = ['w'] * numsets
        text_colors = ['k'] * numsets
        box_trans = [0.7] * numsets
        z_mask = zscores_this > 5
        for int_num,z_out in enumerate(z_mask):
            if z_out:
                bar_colors[int_num] = 'k'
                text_colors[int_num] = 'r'
                box_trans[int_num] = 0
        jeffax[i].bar(np.arange(numsets)-width,distances_this,2*width,alpha=1,color=bar_colors)
        
        jeffax[i].set_ylim(distance_min,distance_max)
        jeff_yticks = jeffax[1].yaxis.get_majorticklocs()
        
        lab_label_pos = 0.95*jeff_yticks[-1] + 0.05*jeff_yticks[0]
        
        sample_label_pos = 0.05*jeff_yticks[-1] + 0.95*jeff_yticks[0]
                
        for int_num,zscore in enumerate(zscores_this):
            jeffax[i].text(int_num,sample_label_pos,
                           str(zscore)[0:4],
                           ha='center', va='bottom',
                           color=text_colors[int_num],size=9,
                           bbox=dict(facecolor='w',alpha=box_trans[int_num],lw=0)
                          )
            #if zscore < 5:
            #    jeffax[i].text(int_num,sample_label_pos,str(zscore)[0:4],ha='center', va='bottom',color="k",size=9)
            #else:
            #    jeffax[i].text(int_num,sample_label_pos,str(zscore)[0:4],ha='center', va='bottom',color="r",size=9)
        jeffax[i].set_xlim(-0.5,numsets-0.5)
        
        #jeffax[i].set_yticks(distance_ticks)
        hist_norm,bins = np.histogram(distances_this,range=(min_bin,max_bin), bins=num_bins,density=True)
        #print hist_norm.shape
        #print bins[:-1].shape
        lognorm_dist = sp.stats.lognorm.pdf(bins,dist_std,scale=dist_scale)
        #hist_norm = hist/binsize / float(hist.sum())
        #print bins[1:]
        #print hist
        #print hist_norm
        #print lognorm_dist
        #pdfax[i].plot(bins[:-1],hist_norm,'r',bins,lognorm_dist,'k')
        pdfax[i].plot(bins,lognorm_dist,'k')
        pdfax[i].bar(bins[:-1]-0.5*binsize,hist_norm,binsize,color='r')
        #pdfax[i].set_yticks(pdf_yticks)
        #print i
        #print pdfax[i]
        
    for i in range(numcols):
        jeffax[bottomplot+i*numrows].set_xlim(-0.5,numsets-0.5)
        jeffax[bottomplot+i*numrows].set_ylim(distance_min,distance_max)
        jeffax[bottomplot+i*numrows].set_xticks(np.arange(numsets))
        jeffax[bottomplot+i*numrows].yaxis.set_major_locator(jeff_locator)
        jeffax[bottomplot+i*numrows].set_xticklabels(Labels,rotation='vertical')
        pdfax[bottomplot+i*numrows].set_xlim(distance_min,distance_max)
        pdfax[bottomplot+i*numrows].yaxis.set_major_locator(pdf_locator)
    
    
    pdf_yticks = pdfax[1].yaxis.get_majorticklocs()
    lab_label_pdf_y_pos = 0.90*pdf_yticks[-1] + 0.10*pdf_yticks[0]
    lab_label_pdf_x_pos = 0.95*distance_min + 0.05*distance_max
    lab_label_jeff_y_pos =  0.1*distance_min + 0.9*distance_max
    for i in range(numspecs):
        jeffax[i].text(0,lab_label_jeff_y_pos,Spectrum_names[i],ha='left', va='top')
        pdfax[i].text(lab_label_pdf_x_pos,lab_label_pdf_y_pos,Spectrum_names[i],ha='left', va='top')
        

    #jefffig.savefig('distance_z_scores.png',bbox_inches='tight',facecolor=[0,0,0,0])
    #pdffig.savefig('lognormal_pdfs.png',bbox_inches='tight',facecolor=[0,0,0,0])

    return zscores
def zscore_principal_components(zscores,positional_array,Spectrum_names,Labels):
    """Performs principal components analysis on the Z scores.
    
    :param zscores: The sample-level Z scores
    :param positional_array: The array defining the structure of the data matrix
    :param Spectrum_names: The list of sample names
    :param Labels: The list of laboratory identifiers
    :returns: zscorepca, the principal components within the Z score space
    """
    numsets = positional_array.shape[0]
    numspecs = positional_array[0,1]
    
    distance_scores = np.reshape(zscores,(numspecs,numsets))
    
    zscorepca = skdecomp.PCA()
    zscorepca.fit(distance_scores.T)
    #zscorepca.fit(np.log(distance_scores.T))
    
    return zscorepca

def zscore_heatmap_plot(zscores,positional_array,Spectrum_names,Labels):
    numsets = positional_array.shape[0]
    numspecs = positional_array[0,1]
    
    distance_scores = np.reshape(zscores,(numspecs,numsets))
    
    plotsize = 8
    
    #Create the figure
    distfig,distax = plt.subplots(figsize=(plotsize*2,plotsize))
    #Plot the distance scores
    
    zscore_max = math.ceil(zscores.max())
    
    im = distax.imshow(distance_scores.T,interpolation="none",cmap="Reds",vmin=0,vmax=zscore_max)
    distbar = distfig.colorbar(im,ax=distax)#,orientation='horizontal')#,label='Distance Score')
    distax.set_yticks(range(numsets))
    distax.set_yticklabels( Labels )
    distax.set_xticks(range(numspecs))
    distax.set_xticklabels(Spectrum_names,rotation='vertical')
    
    distax.set_xlabel('Sample ID',size=15)
    distax.set_ylabel('Data set ID',size=15)
    distbar.set_label('Z Score',size=15)
    return distfig
def zscorelog_heatmap_plot(zscores,positional_array,Spectrum_names,Labels):
    numsets = positional_array.shape[0]
    numspecs = positional_array[0,1]
    
    distance_scores = np.reshape(zscores,(numspecs,numsets))
    
    plotsize = 8
    
    #Create the figure
    distfig,distax = plt.subplots(figsize=(plotsize*2,plotsize))
    #Plot the distance scores
    im = distax.imshow(distance_scores.T,interpolation="none",cmap="seismic",vmin=-np.log(9),vmax=np.log(9))
    distbar = distfig.colorbar(im,ax=distax)#,orientation='horizontal')#,label='Distance Score')
    distax.set_yticks(range(numsets))
    distax.set_yticklabels( Labels )
    distax.set_xticks(range(numspecs))
    distax.set_xticklabels(Spectrum_names,rotation='vertical')
    
    distax.set_xlabel('Sample ID',size=15)
    distax.set_ylabel('Data set ID',size=15)
    distbar.set_label('Z Score',size=15)
    return distfig
def zscore_loadings_plot(zscorepca,positional_array,Spectrum_names,Labels):
#    """Plots the PCA loadings calculated using zscore_principal_components()
#    
#    :param zscorepca: The principal components calculated using zscore_principal_components()
#    :param positional_array: The array defining the structure of the data matrix
#    :param Spectrum_names: The list of sample names
#    :param Labels: The list of laboratory identifiers
#    """
    numsets = positional_array.shape[0]
    numspecs = positional_array[0,1]
    
    plotsize = 8
    loadfig,loadax = plt.subplots(figsize=(plotsize*2,plotsize))
    PCAsign = np.sign(zscorepca.components_[0,:].mean())
    
    #Plot the principal components    
    im = loadax.imshow(zscorepca.components_*PCAsign,interpolation="none",cmap="seismic",vmin=-1,vmax=1)
    loadbar = loadfig.colorbar(im,ax=loadax)#,orientation='horizontal')
    loadax.set_yticks(range(numsets))
    loadax.set_yticklabels( range(1,numsets+1) )
    loadax.set_xticks(range(numspecs))
    loadax.set_xticklabels(Spectrum_names,rotation='vertical')
    
    loadax.set_xlabel('Sample ID',size=15)
    loadax.set_ylabel('Principal Component Number',size=15)
    loadbar.set_label('Component Value',size=15)
    
    for component_number in range(numsets):
        componentscore = "%5.2f" % (zscorepca.explained_variance_ratio_[component_number] * 100)
        componentscore_string = componentscore + " %"
        #loadax.text(0,component_number,componentscore_string,ha='center',va='center',bbox=dict(facecolor='white'))
        loadax.text(numspecs,component_number,componentscore_string,ha='center',va='center',bbox=dict(facecolor='white'))
    return loadfig
def zscore_principal_component_plots(zscores,positional_array,Spectrum_names,Labels):
#    """Performs principal components analysis on the Z scores. Plots the Z scores and principal components.
#    """
    numsets = positional_array.shape[0]
    numspecs = positional_array[0,1]
    
    distance_scores = np.reshape(zscores,(numspecs,numsets))
    
    zscorepca = skdecomp.PCA()
    zscorepca.fit(distance_scores.T)
    #zscorepca.fit(np.log(distance_scores.T))
    plotsize = 8
    
    #Create the figure
    distfig,distax = plt.subplots(figsize=(plotsize*2,plotsize))
    loadfig,loadax = plt.subplots(figsize=(plotsize*2,plotsize))
    
    #Plot the distance scores
    im = distax.imshow(distance_scores.T,interpolation="none",cmap="Reds",vmin=0,vmax=9)
    distbar = distfig.colorbar(im,ax=distax)#,orientation='horizontal')#,label='Distance Score')
    distax.set_yticks(range(numsets))
    distax.set_yticklabels( Labels )
    distax.set_xticks(range(numspecs))
    distax.set_xticklabels(Spectrum_names,rotation='vertical')
    
    distax.set_xlabel('Sample ID',size=15)
    distax.set_ylabel('Data set ID',size=15)
    distbar.set_label('Z Score',size=15)
    
    
    #distfig.savefig('syn_distances.png',bbox_inches='tight',facecolor=[0,0,0,0])
    
    
    PCAsign = np.sign(zscorepca.components_[0,:].mean())
    
    #Plot the principal components    
    im = loadax.imshow(zscorepca.components_*PCAsign,interpolation="none",cmap="seismic",vmin=-1,vmax=1)
    loadbar = loadfig.colorbar(im,ax=loadax)#,orientation='horizontal')
    loadax.set_yticks(range(numsets))
    loadax.set_yticklabels( range(1,numsets+1) )
    loadax.set_xticks(range(numspecs))
    loadax.set_xticklabels(Spectrum_names,rotation='vertical')
    
    loadax.set_xlabel('Sample ID',size=15)
    loadax.set_ylabel('Principal Component Number',size=15)
    loadbar.set_label('Component Value',size=15)
    
    for component_number in range(numsets):
        componentscore = "%5.2f" % (zscorepca.explained_variance_ratio_[component_number] * 100)
        componentscore_string = componentscore + " %"
        loadax.text(numspecs,component_number,componentscore_string,ha='center',va='center',bbox=dict(facecolor='white'))
        
    #loadfig.savefig('syn_loadings.png',bbox_inches='tight',facecolor=[0,0,0,0])
    return zscorepca
def PCAplots(zscorepca,zscores,positional_array,Labels,number_components,axis_inset_position,axis_inset_xlim,axis_inset_ylim):
    numsets = positional_array.shape[0]
    numspecs = positional_array[0,1]
    
    plotsize = 8
    
    PCAsign = np.sign(zscorepca.components_[0,:].mean())
    distance_scores = np.reshape(zscores,(numspecs,numsets))
    zscores_projected = np.dot(zscorepca.components_*PCAsign,distance_scores)
    
    component1score = "%5.2f" % (zscorepca.explained_variance_ratio_[0] * 100)
    xaxislabel = 'Component 1: ' + component1score + " %"
    
    for component_number in range(1,number_components):#numsets):
        #Create the PCA plots
        #print component_number
        scorefig,scoreax = plt.subplots(figsize=(plotsize,plotsize))
        scoreax.scatter(zscores_projected[0,:],zscores_projected[component_number,:])
        
        componentyscore = "%5.2f" % (zscorepca.explained_variance_ratio_[component_number] * 100)
        yaxislabel = 'Component ' + str(component_number+1) + ': '  + componentyscore + " %"
        scoreax.set_xlabel(xaxislabel,size=15)
        scoreax.set_ylabel(yaxislabel,size=15)
        scoreax.set_xlim(0,30)
        
        #print axis_inset_position[component_number][0]
        #print axis_inset_position[component_number][0] is not None
        
        if axis_inset_position[component_number][0] is not None:#component_number < 3:
            axinset = scorefig.add_axes(axis_inset_position[component_number] )
            axinset.scatter(zscores_projected[0,:],zscores_projected[component_number,:])
            xlim = axis_inset_xlim[component_number]
            ylim = axis_inset_ylim[component_number]
            axinset.set_xlim(xlim)
            axinset.set_ylim(ylim)
            for j,labeltext in enumerate(Labels):
                if zscores_projected[0,j] > xlim[0] and zscores_projected[0,j] < xlim[1] and zscores_projected[component_number,j] < ylim[1] and zscores_projected[component_number,j] > ylim[0]:
                    axinset.text(zscores_projected[0,j]+0.1,zscores_projected[component_number,j],
                                 labeltext,
                                 ha='left',va='top',
                                 size='12',
                                 bbox = dict(facecolor='w'))
                else:
                    scoreax.text(zscores_projected[0,j]+0.3,zscores_projected[component_number,j]-0.1,
                                 labeltext,
                                 ha='left',va='top',
                                 size='12',
                                 bbox = dict(facecolor='w'))
        else:
            for j,labeltext in enumerate(Labels):
                scoreax.text(zscores_projected[0,j]+0.3,zscores_projected[component_number,j]-0.1,
                             labeltext,
                             ha='left',va='top',
                             size='12',
                             bbox = dict(facecolor='w'))
        filename = 'pca_component'+str(component_number+1)+'.png'
        scorefig.savefig(filename,bbox_inches='tight',facecolor=[0,0,0,0])
    return
def zscore_outliers(zscorepca,zscores,support_fraction,positional_array,lab_labels,distribution):
    """
    :param zscorepca: The principal components calculated using zscore_principal_components()
    :param zscores: The sample-level Z scores
    :param support_fraction: The fraction of samples that must be retained
    :param positional_array: The array defining the structure of the data matrix
    :param lab_labels: The list of laboratory identifiers
    :param distribution: The distribution assumed when assigning Z scores to each measurement of a sample.
    :returns: z_mask, a Boolean mask that will remove the outliers from the data
    :returns: zscores_projected
    :returns: robust_pdf_params, the parameters of the final distribution used to identify the outliers
    """
    numsets = positional_array.shape[0]
    numspecs = positional_array[0,1]
    
    #len(support_fractions)
    #support_fraction = 0.6
    
    PCAsign = np.sign(zscorepca.components_[0,:].mean())
    distance_scores = np.reshape(zscores,(numspecs,numsets))
    #zscores_projected = zscorepca.transform(distance_scores.T)*PCAsign
    #zscores_projected = np.exp(zscorepca.transform(np.log(distance_scores.T)*PCAsign))
    
    zscores_projected = np.dot(zscorepca.components_*PCAsign,distance_scores).T
    #zscores_projected = np.exp(np.dot(zscorepca.components_*PCAsign,np.log(distance_scores)).T)
    
    #Find those zscore principal components that explain more than 1% of the variance
    #The robust determinant will include all of those
    zscores_mask = zscorepca.explained_variance_ratio_ > 0.01
    
    #Force the second principal component to be active
    zscores_mask[1] = True
    
    #Find the number of active components
    num_active_components = np.count_nonzero(zscores_mask)
    
    #Compute the distance to the origin of each lab to the origin, using only the major components
    norms = np.linalg.norm(zscores_projected[:,0:num_active_components],axis=1)
    #print norms
    #Fit the empirical lognormal distribution to the data
    ln_norms = np.log(norms)
    norm_mean = ln_norms.mean()
    norm_scale = math.exp(norm_mean)
    norm_std = ln_norms.std()
    
    params = distribution.fit(norms,floc=0)
    rv = distribution(*params)
    std = distribution(1)
    
    #lognormal_dist = sp.stats.lognorm(norm_std,scale=norm_scale)
    # Get Z scores for this distribution
    lognormal_int = lognormal_integral(norms,norm_std,norm_mean)
    distance_z = lognormal_z(lognormal_int)
    
    lognormal_int = rv.cdf(norms)
    distance_z = std.ppf(lognormal_int)
    
    #Statistical calculations will be done on the robust distribution, so start with the robust distribution 
    #being the same as the empirical distribution
    norm_mean_robust = norm_mean
    norm_scale_robust = norm_scale
    norm_std_robust = norm_std
    
    z_mask = np.ones_like(distance_z,dtype=bool) #Set the mask to a True array of the same size as the distance array
    num_inliers = np.count_nonzero(z_mask)
    
    #Maximum number of outliers
    max_num_outliers = int((1-support_fraction)*numsets)
    params_robust = params
    
    
    #Robust lognormal distribution
    for i in range(max_num_outliers): # Do this computation for the number of outliers that we expect
        #if np.count_nonzero(z_mask) <= support_fraction*10 : #But stop the computation if there are too many outliers
            #    print i
            #    print z_mask
            #    break
        z_max = distance_z[z_mask].max() # Find the maximum Z score among the remaining inliers. This is the candidate outlier.
        
        if z_max < std.ppf(0.9495): # Check if the candidate outlier's Z score is within the current 95% confidence interval 
            #of the current robust distribution. Stop the computation
            #print i
            #print z_mask
            break #
        
        z_mask = distance_z < z_max #Remove all points with larger Z scores than the current outlier

        #Calculate the new robust distribution
        norms_robust = norms[z_mask]
        ln_norms = np.log(norms_robust)
        norm_mean_robust = ln_norms.mean()
        norm_scale_robust = math.exp(norm_mean_robust)
        norm_std_robust = ln_norms.std()
        
        lognormal_int = lognormal_integral(norms,norm_std_robust,norm_mean_robust)
        distance_z = lognormal_z(lognormal_int)
        
        params_robust = distribution.fit(norms_robust,floc=0)
        rv_robust = distribution(*params_robust)
        lognormal_int = rv_robust.cdf(norms)
        distance_z = std.ppf(lognormal_int)

    robust_pdf_params = params_robust#(norm_std_robust,norm_mean_robust)
    return z_mask,zscores_projected,robust_pdf_params

    
    
    
def zscore_outlier_plots(z_mask,zscores_projected,zscorepca,robust_pdf_params,positional_array,lab_labels,distribution,y_component=1):
    """Plots the principal component scores for each lab along with the final distribution used to calculate the outliers
    
    :param z_mask: a Boolean mask that will remove the outliers from the data
    :param zscores_projected: 
    :param robust_pdf_params: The parameters of the final distribution used to identify the outliers
    :param positional_array: The array defining the structure of the data matrix
    :param lab_labels: The list of laboratory identifiers
    :param distribution: The distribution assumed when assigning Z scores to each measurement of a sample.
    """
    num_plots = 1
    
    #norm_std_robust,norm_mean_robust = robust_pdf_params
    #If specified, take the logarithm of the first component (it will always be positive)
    if take_log:
        zscores_projected[:,0] = np.log(zscores_projected[:,0])
    #Create the first PCA plot to find out how big the contour plot needs to be
    pcaplot,pca_ax = plt.subplots(1,1,figsize = (10,5),sharex=True,sharey=True)
    #Plot the data to see how big the pdfs need to be
    #scatterplot = pca_ax.scatter(zscores_projected[:,0],zscores_projected[:,1])
    scatterplot = pca_ax.scatter(zscores_projected[z_mask,0],zscores_projected[z_mask,y_component],
                                 c='r',edgecolors='none',s=60)
    scatterplot = pca_ax.scatter(zscores_projected[np.invert(z_mask),0],zscores_projected[np.invert(z_mask),y_component],
                                 c='b',edgecolors='none',s=60)
    
    (xmin,xmax) = pca_ax.get_xlim()
    (ymin,ymax) = pca_ax.get_ylim()
    
    #if take_log is False:
    #    xmin = 0
    #ymin = 0

    #Create the plot array
    #pcaplot,pca_ax = plt.subplots(num_plots,1,figsize = (10,5),sharex=True,sharey=True)
    

    #Number of points to use to calculate the MH distance contours
    numgrid = 60
    
    xstep = (xmax-xmin)/numgrid
    ystep = (ymax-ymin)/numgrid
    
    xdata = np.arange(xmin,xmax,xstep)
    ydata = np.arange(ymin,ymax,ystep)
    
    #print xdata
    #print ydata
    
    #Set up the grid mesh for the contour plots
    xx,yy = np.meshgrid(xdata,ydata)
    
    #Calculate the distance function for the contour plots
    dist = np.sqrt(xx ** 2 + yy ** 2)
    if take_log:
        dist = np.sqrt(np.exp(xx) ** 2 + yy ** 2)
    
    #Calculate the lognormal Z function w.r.t. the distance from the origin
    #integrals = lognormal_integral(dist,norm_std_robust,norm_mean_robust)
    #pdf_for_plot = lognormal_z(integrals)
    
    rv = distribution(*robust_pdf_params)
    std = distribution(1)
    integrals = rv.cdf(dist)
    pdf_for_plot = std.ppf(integrals)
    
    #pdf_for_plot = qmet.lognormal_z(dist)
    #pdf_for_plot = lognormal_dist.cdf(dist)
    #pdf_for_plot = pdf_for_plot/pdf_for_plot.max()
    
    #Level curves of the lognormal Z function
    levelsc = [1, 2, 3, 4, 5]
    levelsf = [0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    
    levelsc = list(np.arange(1,std.ppf(0.95))) + [std.ppf(0.95)]
    levelsf = [0.25] + list(np.arange(0.5,std.ppf(0.95),0.5)) + [std.ppf(0.95)]
    
    cformats =  ['{:1.0f}'] * (len(levelsc) - 1)
    cformats += ['{:3.1f}']
    
    cformats_dict = {x:fmt_str.format(x) for x,fmt_str in zip(levelsc,cformats) }
    ccolors = ['w'] * int(len(levelsc)/2)
    ccolors += ['k'] * int( (len(levelsc) + 1)/2)
    
    #print levelsc
    #print cformats_dict
    #print ccolors

    
    #pdf_for_plot = np.sqrt(np.log(pdf_for_plot) * -2)
    
    
    #print pdf_for_plot
    
    ctr1 = pca_ax.contourf(xx,yy,pdf_for_plot,extend='min',cmap='Greys_r',levels=levelsf)
    #ctr2 = pca_ax.contour(xx,yy,pdf_for_plot,extend='neither',cmap='Blues',levels=levelsc)
    #ctr2 = pca_ax.contour(xx,yy,pdf_for_plot,extend='neither',colors=('w','w','k','k','k'),levels=levelsc)
    ctr2 = pca_ax.contour(xx,yy,pdf_for_plot,extend='neither',colors=ccolors,levels=levelsc)
    #plt.colorbar(ctr1,ax=pca_ax)
    #contourlabels = plt.clabel(ctr2,fmt='%1.0f',colors=('w','w','k','k','k'))
    contourlabels = plt.clabel(ctr2,fmt=cformats_dict,colors=ccolors)#('w','w','k','k','k'))
    scatterplot = pca_ax.scatter(zscores_projected[z_mask,0],zscores_projected[z_mask,y_component],
                                 c='r',edgecolors='none',s=60)
    #scatterplot = pca_ax.scatter(zscores_projected[np.invert(z_mask),0],zscores_projected[np.invert(z_mask),y_component],
    #                             c='b',edgecolors='none')
    #scatterplot = pca_ax.scatter(np.log(zscores_projected[z_mask,0]),zscores_projected[z_mask,1],c='r',edgecolors='none')
    #scatterplot = pca_ax.scatter(np.log(zscores_projected[np.invert(z_mask),0]),zscores_projected[np.invert(z_mask),1],c='b',edgecolors='none')
    pca_ax.set_xlim(xmin,xmax)
    pca_ax.set_ylim(ymin,ymax)
    
    component1score = "%5.2f" % (zscorepca.explained_variance_ratio_[0] * 100)
    component2score = "%5.2f" % (zscorepca.explained_variance_ratio_[y_component] * 100)
    xaxislabel = 'Component 1: ' + component1score + " %"
    yaxislabel = 'Component ' + str(y_component + 1) +': ' + component2score + " %"
    
    pca_ax.set_xlabel(xaxislabel,size=20)
    pca_ax.set_ylabel(yaxislabel,size=20)
    outlier_x_text_position = 0.2*xmin + 0.8*xmax
    #alpha = 0.5
    #outlier_y_text_position = (1 - alpha)*ymin + alpha*ymax
    for j,labeltext in enumerate(lab_labels):
        if not(z_mask[j]):
            pca_ax.text(zscores_projected[j,0]+1.0,zscores_projected[j,y_component],
                        labeltext,
                        ha='left',va='center',
                        size='9',
                        bbox = dict(facecolor='w'))
    #        pca_ax.text(outlier_x_text_position,outlier_y_text_position,
    #                    labeltext,
    #                    ha='left',va='top',
    #                    size='9')
    #        alpha = alpha - 0.1
    #        outlier_y_text_position = (1 - alpha)*ymin + alpha*ymax
    return pcaplot
def zscore_projected_plot(zscoreax,zscorepca,zscores_projected,z_mask,robust_pdf_params,positional_array,lab_labels,distribution):
    """Plots the projected statistical distances and the corresponding lab-level Z scores 
    
    :param zscoreax: The axis that the statistical distances will be plotted on
    :param z_mask: a Boolean mask that will remove the outliers from the data
    :param zscores_projected: 
    :param robust_pdf_params: The parameters of the final distribution used to identify the outliers
    :param positional_array: The array defining the structure of the data matrix
    :param lab_labels: The list of laboratory identifiers
    :param distribution: The distribution assumed when assigning Z scores to each measurement of a sample.
    """
    numsets = positional_array.shape[0]
    numspecs = positional_array[0,1]
    
    #Specify the active components as those that explain more than 1% of the total variance
    zscores_mask = zscorepca.explained_variance_ratio_ > 0.01
    zscores_mask[1] = True # Force component 2 to be active
    num_active_components = np.count_nonzero(zscores_mask)
    
    #Calculate the norms of the projected z-score vectors
    norms = np.linalg.norm(zscores_projected[:,0:num_active_components],axis=1)
    
    #Recover the robust PDF of the projected z-score distribution and calculate the aggregate z-scores
    #(norm_std_robust,norm_mean_robust) = robust_pdf_params
    #distribution = sp.stats.lognorm
    rv = distribution(*robust_pdf_params)
    std = distribution(1)
    
    quantile = rv.cdf(norms)
    #quantile[quantile > 0.99999999] = 0.99999999
    #print quantile
    
    distance_z = std.ppf(quantile)#rv.cdf(norms))
    
    if distribution is sp.stats.lognorm:
        norm_mean_robust = robust_pdf_params[2]
        norm_std_robust = robust_pdf_params[0]
        distance_z = np.exp( (np.log(norms) - np.log(norm_mean_robust)) / norm_std_robust)
    
    #qprintstr = '{:4.5f}'#'{' + ':3.1f ' * numsets + '}'
    #print qprintstr
    #print robust_pdf_params
    #print " ".join(qprintstr.format(q) for q in quantile)
    #print qprintstr.format(tuple(map(tuple,quantile)))
    
    #
    
    width = 0.4
    
    #Default bar colors and text box transparency
    bar_colors = ['w'] * numsets
    bar_edge = ['k'] * numsets
    box_trans = [0.7] * numsets
    for int_num,z_out in enumerate(z_mask):
        #Color and transparency if the point is an outlier
        if not(z_out):
            bar_colors[int_num] = 'r'
            bar_edge[int_num] = 'r'
            box_trans[int_num] = 0
    #zscoreax.bar(np.arange(numsets)-width,norms,2*width,alpha=1,color=bar_colors)
    zscoreax.bar(np.arange(numsets),norms,2*width,alpha=1,color=bar_colors,edgecolor=bar_edge)
    
    (ymin,ymax) = zscoreax.get_ylim()
    
    sample_label_pos = 0.05*ymax + 0.95*ymin
    
    for int_num,zscore in enumerate(distance_z):
        if(z_mask[int_num]): 
            label_pos = sample_label_pos + (norms[int_num])
        else:
            label_pos = sample_label_pos
    
        zscoreax.text(int_num,label_pos,
                      str(zscore)[0:4],
                      ha='center', va='bottom',
                      color='k',size=9,
                      bbox=dict(facecolor='w',alpha=box_trans[int_num],lw=0)
                     )
    zscoreax.set_xlim(-0.5,numsets-0.5)
    zscoreax.set_xticks(np.arange(numsets))
    zscoreax.set_xticklabels(lab_labels,rotation='vertical')
    return
def zscore_loadings_bar(loadax,zscorepca,positional_array,Spectrum_names,lab_labels):
    """Plots the PCA loadings calculated using zscore_principal_components()
    
    :param zscoreax: The axis that the PCA loadings will be plotted on
    :param zscorepca: The principal components calculated using zscore_principal_components()
    :param positional_array: The array defining the structure of the data matrix
    :param Spectrum_names: The list of sample names
    :param lab_labels: The list of laboratory identifiers
    """
    numsets = positional_array.shape[0]
    numspecs = positional_array[0,1]
    
    #width = 5
    #loadfig,loadax = plt.subplots(figsize=(numcols*width,numrows+1),sharex=True,sharey=True)
    PCAsign = np.sign(zscorepca.components_[0,:].mean())
    
    zscores_mask = zscorepca.explained_variance_ratio_ > 0.01
    zscores_mask[1] = True
    num_active_components = np.count_nonzero(zscores_mask)
    
    #Calculate the positions where the bars will go
    total_width = 0.4
    width = total_width/num_active_components
    positions = np.arange(-total_width,total_width,2*width)
    
    #Create the spectrum of colors that will be used for the plot
    bar_properties = np.arange(1,0,-1.0/num_active_components)
    
    #bar_colors = ['w'] * numsets
    bar_colors = ['w'] * numsets
    text_colors = ['k'] * numsets
    box_trans = [0.7] * numsets
    #Plot the principal components
    for component_num in range(num_active_components):
        #intensity = bar_properties[component_num]
        intensity = 1 - zscorepca.explained_variance_ratio_[component_num]
        #bar_colors = [[intensity] * 2 + [1]] * numsets
        bar_colors = [[intensity] * 3] * numsets
        position = positions[component_num]
        im = loadax.bar(np.arange(numspecs)+position,zscorepca.components_[component_num,:]*PCAsign,
                        2*width,alpha=1,color=bar_colors,edgecolor='k')
    #bar_colors = ['b'] * numsets
    #im = loadax.bar(np.arange(numspecs)-width,zscorepca.components_[1,:]*PCAsign,2*width,alpha=0.7,color=bar_colors)
    #bar_colors = ['c'] * numsets
    #im = loadax.bar(np.arange(numspecs)-width,zscorepca.components_[2,:]*PCAsign,2*width,alpha=0.3,color=bar_colors)
    #im = loadax.imshow(zscorepca.components_*PCAsign,interpolation="none",cmap="seismic",vmin=-1,vmax=1)
    #loadbar = loadfig.colorbar(im,ax=loadax)#,orientation='horizontal')
    loadax.set_xlim(-1,numspecs + 2)
    loadax.set_xticks(range(numspecs))
    loadax.set_xticklabels(Spectrum_names,rotation='vertical')
    (ymin,ymax) = loadax.get_ylim()
    
    #loadax.set_xlabel('Sample ID',size=15)
    #loadax.set_xlabel('Principal Component Number',size=15)
    #loadbar.set_label('Component Value',size=15)
    
    center = (ymax+ymin)/2.0
    interval = (ymin-ymax)/(num_active_components + 1)
    positions = np.arange(ymax+interval,ymin,interval)
    
    for component_number in range(num_active_components):
        componenet_label_position = positions[component_number]
        componentscore = "%5.2f" % (zscorepca.explained_variance_ratio_[component_number] * 100)
        componentscore_string = componentscore + " %"
        loadax.text(numspecs + 1,componenet_label_position,componentscore_string,ha='center',va='center')#,bbox=dict(facecolor='white'))
        #loadax.text(numspecs,component_number,componentscore_string,ha='center',va='center',bbox=dict(facecolor='white'))
    return
class Project(object):
    """The top-level project class for the interlaboratory comparison module
    
    :param distance_metric_dict: The dictionary of distance metrics that will be used to detect outliers. Outliers will be detected separately for each distance function
    :param Sample_names: The list of sample names. Each sample must have its own name
    :param Data_set_names: The list of data sets in the interlaboratory comparison
    :param x_data_list: The list of x data in the data array. For NMR, this will be an array of chemical shifts
    :param range_to_use: Which data from the spectrum will actually be used. If None, do not remove any data
    :param distribution_function: Which distribution will be assumed when assigning Z scores to each measurement of a sample. The default is sp.stats.lognorm
    :param outlier_dist: Which distribution will be assumed when detecting outliers. The default is the same as distribution_function
    
    """
    def __init__(self,
                 distance_metric_dict=None,
                 #distance_metric_function_list=None,
                 Sample_names=None,
                 Data_set_names=None,
                 x_data_list=None,
                 range_to_use=None,
                 distribution_function=sp.stats.lognorm,
                 outlier_dist=None):
        
        # Feature data and corresponding bin name
        self._data = None
        self._x_data = np.array(x_data_list)
        
        # Names for the data sets (or laboratories) and samples ()
        self.Labels = Data_set_names
        self.Spectrum_names = Sample_names
        
        #Array that defines the positions of the arrays
        self._positional_array = positional_array(len(self.Spectrum_names),len(self.Labels))
        
        #Range of data to use for analysis
        self._range_to_use = range_to_use
        
        self.distance_metrics = distance_metric_dict
        self.distance_matrix = dict()
        self.distance_matrix_bylab = dict()
        self.distances = dict()
        self.mahalanobis_dict = None
        
        
        self.zscores = dict()
        self.pca = dict()
        self.projected_zscores = dict()
        self.outlier_mask = dict()
        self.zscore_pdf = dict()
        self.pdfs = dict()
        self._distribution_function = distribution_function
        self._outlier_distribution = outlier_dist
        if outlier_dist is  None:
            self._outlier_distribution = distribution_function
        return
    
    @property
    def data(self):
        """
        :param data_file: The data file that is passed to qmet.fanndataread
        :type data_file: str
        """
        return self._data
    
    @data.setter
    def data(self,data_file):
        """
        :param data_file: The data file that is passed to qmet.fanndataread
        :type data_file: str
        """
        features,classes,fileinfo = fanndataread(data_file)
        if self._range_to_use is not None: 
            features = features[:,self._range_to_use]
            self._x_data = self._x_data[self._range_to_use]
        features = fix_spectrum(features)
        features = np.transpose(features.T/features.sum(axis=1))
        self._data = features
        return
    
    def plot_data(self):
        max_bin = self.data.max()
        plot_increment = self.data.shape[1]/3.0
        spectrumplot(self.data,increment=plot_increment,vertsize=10,maxz=1/35.0,symmetric=False,xdata=-1*self._x_data)
    
    def set_distances(self):
        """Calculates the interspectral distances for each metric
        """
        for distance_metric in self.distance_metrics:
            metric = distance_metric['metric']
            metric_function = distance_metric['function']
            self.set_distance_matrix(metric,metric_function)
    
    def set_distance_matrix(self,metric,metric_function=None):
        if metric_function == None: metric_function = metric
        self.distance_matrix_bylab[metric],self.distance_matrix[metric] = distances(self._data,
                                                                                      metric_function,
                                                                                      self._positional_array,
                                                                                      self.mahalanobis_dict)
        return
    
    def process_mahalanobis(self,threshold=1e-1):
        """Calculates the Mahalanobis distance
        """
        self.mahalanobis_dict = mahalanobis_covariance(self._data,self._positional_array,threshold)
    
    def set_zscores(self):
        """Calculates the sample-level Z scores for each metric
        """
        for distance_metric in self.distance_metrics:
            metric = distance_metric['metric']
            self.set_zscore_matrix(metric)
        
    def set_zscore_matrix(self,metric):
        self.zscores[metric],self.distances[metric],self.pdfs[metric] = get_zscores(
            self.distance_matrix[metric],self._positional_array,self._distribution_function
                                                                                   )
        return
    
    def set_zscore_principal_components(self):
        """Performs the principal components analysis on the sample-level Z scores
        """
        for distance_metric in self.distance_metrics:
            metric = distance_metric['metric']
            self.set_pca(metric)
    
    def set_pca(self,metric):
        self.pca[metric] = zscore_principal_components(
            self.zscores[metric],self._positional_array,self.Spectrum_names,self.Labels
        )
    
    def find_all_outliers(self):
        """Finds the laboratory-outliers for each metric
        """
        for distance_metric in self.distance_metrics:
            metric = distance_metric['metric']
            self.find_outliers(metric)
        
    def find_outliers(self,metric):
        support_fraction = 0.6
        self.outlier_mask[metric],self.projected_zscores[metric],self.zscore_pdf[metric] = zscore_outliers(
            self.pca[metric],self.zscores[metric],support_fraction,self._positional_array,self.Labels,self._outlier_distribution
            )
        
    
    def plot_distance_fig(self,plot_range=None,cmap='Greys',linecolor='k',distance_metrics=None):
        """For each sample, generates the following plots:
        * A plot of the spectra generated for that sample by each laboratory
        * For each metric, a heat map plot of the interspectral distance matrix
        
        :key plot_range: An iterable of integers specifying which sample labels to plot
        :key cmap: The color map that will be used for the distance heat maps
        :key linecolor: The line color that will be used for the spectral data
        :key distance_metrics: A list of the distance metrics for which heat maps will be plotted. If None, plot heat maps for all metrics in this project
        :returns: distance_fig, the distance measure figure matplotlib object.
        """
        
        if distance_metrics is None: distance_metrics = [metric['metric'] for metric in self.distance_metrics]
        
        distance_matrix_list = [self.distance_matrix[metric] for metric in distance_metrics]
        #distance_name_list = [metric['metric'] for metric in self.distance_metrics]
        #distance_matrix_list = [self.distance_matrix[metric] for metric in distance_name_list]
        
        if plot_range is None: plot_range = range(self._positional_array[0,1])
        
        distance_fig = distance_measure_fig(self._x_data,self._data,
                                            self._positional_array,
                                            self.Spectrum_names,plot_range,
                                            distance_matrix_list,distance_metrics,self.Labels,cmap=cmap,linecolor=linecolor
                                           )
        return distance_fig
    
    def plot_zscore_distances(self,metric,plot_range=None,numcols=2) :
        """Plots a bar chart of the average interspectral distance for each sample, annotated with the generalized Z score for each sample
        
        :param metrics: The metric for which the distances will be plotted
        :key plot_range: An iterable of integers specifying which sample labels to plot
        :key numcols: The number of columns in the distance plot
        :returns zscore_distances_fig: The distances and scores plot as a matplotlib figure object
        """
        zscore_distances_fig = plot_zscores(self.distances[metric],self.zscores[metric],self.pdfs[metric],
                                            self._positional_array,self.Spectrum_names,self.Labels,
                                            num_bins=20,plot_range=plot_range,numcols=numcols,metric=metric
                                           )
        return zscore_distances_fig
    
    def plot_distance_histograms(self,metric,plot_range=None,numcols=2):
        """Plots a histogram of the average interspectral distance for each sample, along with the corresponding fit
        
        :param metrics: The metric for which the distances will be plotted
        :key plot_range: An iterable of integers specifying which sample labels to plot
        :key numcols: The number of columns in the distance plot
        :returns zscore_distances_fig: The distances and scores plot as a matplotlib figure object
        """
        distance_histograms_fig = plot_histograms(self.distances[metric],self.zscores[metric],self.pdfs[metric],
                                                  self._positional_array,self.Spectrum_names,self.Labels,self._distribution_function,
                                                  num_bins=20,plot_range=plot_range,numcols=numcols
                                                 )
        return distance_histograms_fig
        
    def plot_zscore_heatmap(self,metric):
        zscore_heatmap = zscore_heatmap_plot(self.zscores[metric],self._positional_array,self.Spectrum_names,self.Labels)
        return zscore_heatmap
        
    def plot_zscore_loadings_heatmap(self,metric):
        zscore_loadings_heatmap = zscore_loadings_plot(self.pca[metric],self._positional_array,self.Spectrum_names,self.Labels)
        return zscore_loadings_heatmap
    
    def plot_zscore_outliers(self,metric,y_component=1):
        """Plots the principal component scores for each lab along with the final distribution used to calculate the outliers
        
        :param metric: The metric used to calculate the interspectral distances
        :param y_component: Which principal component to use on the Y axis, if not the first
        :returns: zscore_outliers_fig, the Z score outlier plot as a matplotlib figure object
        """
        zscore_outliers_fig = zscore_outlier_plots(self.outlier_mask[metric],self.projected_zscores[metric],
                                                   self.pca[metric],self.zscore_pdf[metric],
                                                   self._positional_array,self.Labels,self._outlier_distribution,y_component=y_component)
        
        zscore_outliers_fig.text(0.85,0.8,metric,ha='right',va='top',size=15)
        return zscore_outliers_fig
    
    def plot_projected_zscores(self,distance_metrics=None):
        """Plots the projected statistical distances annotated with the corresponding laboratory-level Z scores.
        
        :key distance_metrics: A list of the distance metrics for which statistical distances will be plotted. If None, plot statistical distances for all metrics in this project
        :returns: zscorefig, the projected statistical distances plot as a matplotlib figure object
        """
        if distance_metrics is None: 
            distance_metrics = self.distance_metrics
        else:
            metrics_list = []
            for name in distance_metrics:
                metrics_list += [metric for metric in self.distance_metrics if name in metric['metric']]
            distance_metrics = metrics_list
        
        numplots = len(distance_metrics)
        numcols = 1
        numrows = int(math.floor((numplots + numcols - 1)/ numcols))
        width = 5
        zscorefig,zscoreax_array = plt.subplots(numrows,numcols,figsize=(numcols*width,numrows+1),sharex=True,sharey=True)
        if numplots > 1:
            zscoreax = zscoreax_array.flatten('F')
        else:
            zscoreax = np.array([zscoreax_array])
        
        ylabel = r"Projected Statistical Distance, $||T_L||$"
        if numplots < 2:
            ylabel = ylabel.replace(' ','\n')
        
        zscorefig.text(0.00,0.5,ylabel,rotation='vertical',ha='right',va='center',multialignment='center',size=15)
        zscorefig.text(0.5,0.00,'Data set ID',ha='center',va='top',size=15)
        
        
        
        for axis,distance_metric in zip(zscoreax,distance_metrics):
            metric = distance_metric['metric']
            zscore_projected_plot(axis,
                                  self.pca[metric],
                                  self.projected_zscores[metric],
                                  self.outlier_mask[metric],
                                  self.zscore_pdf[metric],
                                  self._positional_array,
                                  self.Labels,self._outlier_distribution)
        for axis,distance_metric in zip(zscoreax,distance_metrics):
            metric = distance_metric['metric']
            (ymin,ymax) = axis.get_ylim()
            metric_label_pos = 0.2*ymin + 0.8*ymax
            #axis.text(0,metric_label_pos,metric,ha='left', va='center')
            axis.set_title(metric)
            #print metric_label_pos
            axis_locator = plt.MaxNLocator(nbins=4,prune='upper')
            axis.yaxis.set_major_locator(axis_locator)
        zscorefig.tight_layout()
        return zscorefig
    def plot_zscore_loadings(self,distance_metrics=None):
        """Plots the principal component loadings for the statistical distances
        
        :key distance_metrics: A list of the distance metrics for which loadings will be plotted. If None, plot loadings for all metrics in this project
        :returns: loadfig, the projected statistical loadings plot as a matplotlib figure object
        """
        
        if distance_metrics is None: 
            distance_metrics = self.distance_metrics
        else:
            metrics_list = []
            for name in distance_metrics:
                metrics_list += [metric for metric in self.distance_metrics if name in metric['metric']]
            distance_metrics = metrics_list
        
        numplots = len(distance_metrics)
        numcols = 1
        numrows = int(math.floor((numplots + numcols - 1)/ numcols))
        width = 5
        loadfig,loadax_array = plt.subplots(numrows,numcols,figsize=(numcols*width,numrows+1),sharex=True,sharey=True)
        if numplots > 1:
            loadax = loadax_array.flatten('F')
        else:
            loadax = np.array([loadax_array])
        loadfig.text(0.00,0.5,u"Component Value",rotation='vertical',ha='center',va='center',size=15)
        loadfig.text(0.5,0.00,'Cluster ID',ha='center',va='top',size=15)
        for axis,distance_metric in zip(loadax,distance_metrics):
            metric = distance_metric['metric']
            zscore_loadings_bar(axis,
                                self.pca[metric],
                                self._positional_array,
                                self.Spectrum_names,
                                self.Labels)
        for axis,distance_metric in zip(loadax,distance_metrics):    
            metric = distance_metric['metric']
            (ymin,ymax) = axis.get_ylim()
            metric_label_pos = 0.9*ymin + 0.1*ymax
            #axis.text(0.5,metric_label_pos,metric,ha='left', va='center')
            axis.set_title(metric)
            #print metric_label_pos
            axis_locator = plt.MaxNLocator(nbins=5,prune='both')
            axis.yaxis.set_major_locator(axis_locator)
        loadfig.tight_layout()
        return loadfig