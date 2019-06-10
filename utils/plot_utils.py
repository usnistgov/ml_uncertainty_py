import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

def make_grid_plot(numrows,numcols,figsize=None,plotsize=None,
                   column_width=6,row_height=4,
                   label_buffers=None,
                   ylabel_buffer=0.75,xlabel_buffer=0.5,rightlabel_buffer=0,
                   xlabel=None,ylabel=None,right_ylabel=None,label_size=15,
                   add_buffer=False,
                   **subplots_args):

    if plotsize is not None:
        column_width,row_height = plotsize
    
    if label_buffers is not None:
        xlabel_buffer,ylabel_buffer = label_buffers
    
    full_width = numcols*column_width
    full_height = numrows*row_height
    
    if figsize is not None:
        full_width,full_height = figsize
    
    if add_buffer:
        full_width = full_width + ylabel_buffer + rightlabel_buffer
        full_height = full_height + xlabel_buffer
    
    bottom_buffer = xlabel_buffer/full_height
    left_buffer = ylabel_buffer/full_width
    right_buffer = rightlabel_buffer/full_width

    ylabel_pos = 0.5*(1+bottom_buffer)
    xlabel_pos = 0.5*(1+left_buffer-right_buffer)
  
    fs = (full_width,full_height)
    #if figsize is not None:
    #    fs = figsize
    fig,axes = plt.subplots(numrows,numcols,figsize=fs,squeeze=False,**subplots_args)
    fig.subplots_adjust(left=left_buffer,right=1-right_buffer,top=1,bottom=bottom_buffer)
    
    if ylabel:
        fig.text(0,ylabel_pos,ylabel,size=label_size,rotation='vertical',va='center',multialignment='center',ha='left')
    if right_ylabel:
        fig.text(1,ylabel_pos,right_ylabel,size=label_size,rotation='vertical',
                 va='center',multialignment='center',ha='right')
    if xlabel:
        fig.text(xlabel_pos,0.0,xlabel,ha="center",va="bottom",multialignment='center',size=label_size)
        
    return fig,axes

class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
class MidpointDeNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y =  [0, 0.5, 1],[self.vmin, self.midpoint, self.vmax]
        return np.ma.masked_array(np.interp(value, x, y))
class LogMidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False, 
                linthresh=1,linscale=1):
        
        mcolors.Normalize.__init__(self, vmin, vmax, clip)
        self.lnmax = np.log(self.vmax)
        self.lnmin = np.log(-self.vmin)
        self.linthresh = linthresh
        #self.linscale = linscale
        self.logthresh = np.log(linthresh)
        self.dec_up = self.lnmax-self.logthresh
        self.dec_dn = self.lnmin-self.logthresh
        self.midpoint = self.dec_dn/(self.dec_up+self.dec_dn)
        total_dec = self.dec_up+self.dec_dn+linscale*np.log(10)
        self.linscale = (linscale*np.log(10))/total_dec/2
        
#         print(self.dec_up)
#         print(self.dec_dn)
#         print(total_dec)
#         print(self.linscale)
    
    def __call__(self,value,clip=None):
        y = np.zeros_like(value)
        
        #Find where value is in the upper or lower log range or the linear range
        log_positive = value>self.linthresh
        log_negative = value<-1*self.linthresh
        linear = ~np.logical_or(log_positive,log_negative)
        
        y[log_positive] = ((np.log(value[log_positive]) - self.logthresh)/(self.dec_up))*(0.5-self.linscale)+(0.5+self.linscale)
        y[log_negative] = (1-(np.log(-1*value[log_negative]) - self.logthresh)/(self.dec_dn))*(0.5-self.linscale)

        y[linear] = ((value[linear] + self.linthresh)/(self.linthresh)-1)*self.linscale+0.5
        
        return np.ma.masked_array(y)
def make_diverging_colormap(map_name,
                            midpoint=0.5,crush_point=0.25,
                            min_color=(0,0,0.5,),
                            crush_lo_color=(0,0,1),
                            mid_color=(1,1,1),
                            crush_hi_color=(1,0,0),
                            max_color=(0.3,0,0)
                 ):
    '''Reproduces seismic colormap by default'''
    crush_lo = midpoint - crush_point
    crush_hi = midpoint + crush_point
    zero = 0
    one = 1
    
    color_positions = (
        (zero,     min_color),
        (crush_lo, crush_lo_color),
        (midpoint, mid_color),
        (crush_hi, crush_hi_color),
        (one,      max_color)
    )
    
    cmap = mcolors.LinearSegmentedColormap.from_list(map_name,list(color_positions))
    return cmap

seismic_with_black = make_diverging_colormap('seismic_with_black',mid_color=(0,0,0))
seismic_with_alpha = make_diverging_colormap('seismic_with_alpha',crush_point=0.05,
                                             mid_color=(1,1,1,1))
RdBu_with_black = make_diverging_colormap('RdBu_with_black',crush_point=0.05,
                                          min_color=(0,0,1),
                                          crush_lo_color=(0,0,0),
                                          crush_hi_color=(0,0,0),
                                          max_color=(1,0,0),
                                          mid_color=(0,0,0))
RdWt = mcolors.LinearSegmentedColormap.from_list('RdWt',[(1,1,1,0),(1,0,0,1)])
greys_with_alpha = mcolors.LinearSegmentedColormap.from_list('greys_with_alpha',[(1,1,1,0),(0,0,0,1)])