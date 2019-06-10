import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.patches as mpatches

def pca_uncertainty_plot(ax,
                         series_colors,
                         classes,
                         scores,
                         bootstrap_classes,
                         bootstrap_scores,
                         class_names=[],
                         x_comp=0,y_comp=1,
                         boot_scatter=False,
                         sample_ellipses=False,
                         boot_group_ellipses=False,
                         group_ellipses=False,
                         sample_bars=False,
                         legend=False,
                         group_tags=False,
                        ):
    #Create the figure
    #fig,axes = plt.subplots(1,2,figsize=(12,6))
    ellipses = []
    F_alpha = 0.05 # 95% confidence limit
    
    for class_num,color in enumerate(series_colors):
        #Get the column that defines the samples as being in this class
        sample_class = classes[:,class_num]
        #Convert to a mask
        class_mask = sample_class == 1        
        #PCA scores
        x_pl = scores[class_mask,x_comp]
        y_pl = scores[class_mask,y_comp]
        ax.scatter(x_pl,y_pl,c=color,edgecolors='k',s=30,zorder=5)
        
        #Number of samples in this class
        num_samples = np.count_nonzero(class_mask)
        
        if bootstrap_classes is not None:
            #Get the column that defines the samples as being in this class
            sample_class = bootstrap_classes[:,class_num]
            #Change to a mask
            class_mask_boot = sample_class == 1
            
            #Bootstrap scores
            x_sc = bootstrap_scores[class_mask_boot,x_comp]
            y_sc = bootstrap_scores[class_mask_boot,y_comp]
        
        #Bootstrap scores scatter plot
        if boot_scatter: ax.scatter(x_sc,y_sc,c=color,edgecolors='none',s=1,zorder=1)
        
        #Error bars on individual samples from bootstrap 
        if sample_bars or sample_ellipses:
            for point in range(num_samples):
                x_pt = x_sc[point::num_samples]
                y_pt = y_sc[point::num_samples]
                if sample_bars:
                    x_err = (np.percentile(x_pt,97.5) - np.percentile(x_pt,2.5))/2
                    y_err = (np.percentile(y_pt,97.5) - np.percentile(y_pt,2.5))/2

                    ax.errorbar(x_pl[point],y_pl[point],xerr=x_err,yerr=y_err,zorder=3,color='k')
                if sample_ellipses:
                    #Calculate the covariance matrix for the bootstrap scores for this sample
                    cov = np.cov(x_pt,y_pt)
                    #Calculate the eigenvalues and eigenvectors of cov
                    lam,v = np.linalg.eig(cov)
                    #Calculate the size and orientation of the confidence ellipse
                    lam = np.sqrt(lam)
                    theta = np.degrees(np.arctan2(*v[:,0][::-1]))
                    
                    pca_samples = len(x_pt)
                    
                    #Calculate the 95% confidence limit based on the F distribution
                    df1 = 2
                    df2 = pca_samples - df1
                    F_val = sp.stats.f.isf(F_alpha,df1,df2)

                    F_mult = F_val * df1*(pca_samples-1)/(df2)
                    F_mult = np.sqrt(F_mult)
                    ellipse_dict = dict(xy=(np.mean(x_pt),np.mean(y_pt)),
                            width=lam[0]*2*F_mult,height=lam[1]*2*F_mult,
                            angle=theta,linewidth=1,linestyle='--',facecolor='none',edgecolor=color,zorder=4)
                    ell = mpatches.Ellipse(**ellipse_dict)
                    if boot_scatter: ell.set_edgecolor('0.2')
                    if boot_group_ellipses: ell.set_edgecolor('0.6')
                    
                    ax.add_artist(ell)
        
        if boot_group_ellipses and np.count_nonzero(class_mask_boot):
            cov = np.cov(x_sc,y_sc)
            lam,v = np.linalg.eig(cov)
            lam = np.sqrt(lam)
            theta = np.degrees(np.arctan2(*v[:,0][::-1]))
            
            pca_samples = len(x_sc)
            
            df1 = 2
            df2 = pca_samples - df1
            F_val = sp.stats.f.isf(F_alpha,df1,df2)

            F_mult = F_val * df1*(pca_samples-1)/(df2)
            F_mult = np.sqrt(F_mult)
            ellipse_dict = dict(xy=(np.mean(x_sc),np.mean(y_sc)),
                                width=lam[0]*2*F_mult,height=lam[1]*2*F_mult,
                                angle=theta,linewidth=2,linestyle='--',facecolor='none',edgecolor=color,zorder=4)
            ell = mpatches.Ellipse(**ellipse_dict)
            if boot_scatter: ell.set_edgecolor('k')
            ax.add_artist(ell)
        
        if group_tags:
            if not boot_group_ellipses:
                cov = np.cov(x_sc,y_sc)
                lam,v = np.linalg.eig(cov)
                lam = np.sqrt(lam)
                theta = np.degrees(np.arctan2(*v[:,0][::-1]))
                pca_samples = len(x_sc)
                df1 = 2
                df2 = pca_samples - df1
                F_val = sp.stats.f.isf(F_alpha,df1,df2)                
                F_mult = F_val * df1*(pca_samples-1)/(df2)
                F_mult = np.sqrt(F_mult)
            x_text_offset = lam[0]*F_mult*-1.1
            y_text_offset = lam[1]*F_mult*-1.1
            ax.text(np.mean(x_sc+x_text_offset),np.mean(y_sc+y_text_offset),
                       class_names[class_num],ha='center',va='center',
                       bbox=dict(edgecolor=color,facecolor='white', alpha=0.9))
            
        if group_ellipses: 
            #Calculate the covariance matrix for the bootstrap scores for this sample
            cov = np.cov(x_pl,y_pl)
            #Calculate the eigenvalues and eigenvectors of cov
            lam,v = np.linalg.eig(cov)
            
            #Calculate the size and orientation of the confidence ellipse
            lam = np.sqrt(lam)
            theta = np.degrees(np.arctan2(*v[:,0][::-1]))
            
            df1 = 2
            df2 = num_samples - df1

            F_val = sp.stats.f.isf(F_alpha,df1,df2)

            F_mult = F_val * df1*(num_samples-1)/(df2)
            F_mult = np.sqrt(F_mult)
            ellipse_dict = dict(xy=(np.mean(x_pl),np.mean(y_pl)),
                                width=lam[0]*2*F_mult,height=lam[1]*2*F_mult,
                                angle=theta,linewidth=2,zorder=2,facecolor='none',edgecolor=color)
            ell = mpatches.Ellipse(**ellipse_dict)
            if boot_scatter: ell.set_edgecolor('k')
            ax.add_artist(ell)
    
    ax.tick_params(labelsize=12)
    if legend:
        patches = []
        for class_name,color in zip(class_names,series_colors):
            patches += [mpatches.Patch(facecolor=color,edgecolor='k',label=class_name)]
        ax.legend(handles=patches)

def score_classes(plsax,
                  series_colors,
                  classes,
                  scores,
                  bootstrap_classes,
                  bootstrap_scores,
                  class_names=[],
                  x_comp=0,y_comp=1,
                  boot_scatter=False,
                  sample_ellipses=False,
                  boot_group_ellipses=False,
                  group_ellipses=False,
                  sample_bars=False,
                  legend=False,
                  f_alphas=None):
    
    if f_alphas is None:
        f_alphas = np.linspace(0.05,0.95,5)
    #print(f_alphas)
    
    F_scores = []
    
    for class_num,color in enumerate(series_colors):
        #Get the column that defines the samples as being in this class
        sample_class = classes[:,class_num]
        #Convert to a mask
        class_mask = sample_class == 1        
        #PCA scores
        x_pl = scores[class_mask,x_comp]
        y_pl = scores[class_mask,y_comp]
        #plsax.scatter(x_pl,y_pl,c=color,edgecolors='k',s=30,zorder=5)
        
        #Number of samples in this class
        num_samples = np.count_nonzero(class_mask)
        
        #Get the column that defines the samples as being in this class
        sample_class = bootstrap_classes[:,class_num]
        #Change to a mask
        class_mask_boot = sample_class == 1
        
        #Bootstrap scores for this class
        x_sc = scores_boot[class_mask_boot,x_comp]
        y_sc = scores_boot[class_mask_boot,y_comp]
        if np.count_nonzero(class_mask_boot):
            #Calculate the covariance matrix of the boot scores so that we can Hoetelling the samples
            cov = np.cov(x_sc,y_sc)
            icov = np.linalg.inv(cov)
            #print(cov)
            pca_samples = len(x_sc)
            lam,v = np.linalg.eig(cov)

            #Calculate the size and orientation of the confidence ellipse
            lam = np.sqrt(lam)
            theta = np.degrees(np.arctan2(*v[:,0][::-1]))

            df1 = 2
            df2 = pca_samples - df1

            Hoetelling_mult = df1*(pca_samples-1)/(df2)

            xy_tuple = (np.mean(x_sc),np.mean(y_sc))

            T_center = np.array([xy_tuple])
            #print(T_center.shape)

            #print(lam)

            for alpha in f_alphas:
                #print(alpha)
                F_val = sp.stats.f.isf(alpha,df1,df2)
                F_mult = F_val * Hoetelling_mult
                F_mult = np.sqrt(F_mult)
                #print(F_mult)

                ellipse_dict = dict(xy=xy_tuple,
                                    width=lam[0]*2*F_mult,height=lam[1]*2*F_mult,
                                    angle=theta,linewidth=1,linestyle=':',zorder=2,facecolor='none',edgecolor=color)
                ell = mpatches.Ellipse(**ellipse_dict)
                plsax.add_artist(ell)

            #Iterate over the sample points and calculate the Hoetelling distance and the corresponding survival function
            for xp,yp in zip(x_pl,y_pl):
                sample_point = np.array([[xp,yp]])
                #print(sample_point.shape)

                t_rel = sample_point - T_center
                dist = np.sqrt(np.dot(t_rel,t_rel.T))
                #print(dist)

                t_right = np.dot(icov,t_rel.T)
                #print(t_right)

                t2_distance = np.dot(t_rel,t_right)
                t2_F = t2_distance / Hoetelling_mult
                F_score = sp.stats.f.sf(t2_F,df1,df2)
                F_scores += [F_score]
            
    F_scores = np.concatenate(F_scores,axis=0)
    return F_scores
def plot_bootstrap_pls(train_data,y_train,test_data=None,y_test=None,axes_row=None,train_colors=None,test_colors=None,group=False):
    
    num_train = len(y_train)
    num_test = 0
    if test_data is not None:
        num_test = len(y_test)
    num_total = num_train + num_test
    
    if train_colors is None:
        train_colors = ['w'] * (num_train)
        train_colors = np.array(train_colors)
        train_colors[y_train > 0] = 'k'
    if test_data is not None:
        if test_colors is None:
            test_colors = ['w'] * (num_test)
            test_colors = np.array(test_colors)
            test_colors[y_test > 0] = 'k'
    
    errbar_dict = dict(fmt='none',ecolor='k',capsize=5,zorder=-100,lw=2)
    
    #Get the median of the bootstrap data
    train_predict = np.median(train_data,axis=1)
    
    #Confidence limits and errorbar widths
    ci_upper_train = np.percentile(train_data,97.5,axis=1)
    ci_lower_train = np.percentile(train_data,2.5,axis=1)
    error_train = (ci_upper_train - ci_lower_train)/2
    
    if test_data is not None:
        test_predict = np.median(test_data,axis=1)
        ci_upper_test = np.percentile(test_data,97.5,axis=1)
        ci_lower_test = np.percentile(test_data,2.5,axis=1)
        error_test = (ci_upper_test - ci_lower_test)/2

    #Class boundary value and line formatting
    cv = plu.estimate_class_boundary(train_predict,y_train)
    class_boundary_dict = dict(color='k',ls=':',lw=2,zorder=-100)
    
    #Pearson R value
    #Pearson R on the test data if available, otherwise on the training data
    if test_data is not None:
        r,p = scipy.stats.pearsonr(y_test,test_predict)
        residual,err,mse = plu.get_residual_stats(y_test,test_predict)
    else:
        r,p = scipy.stats.pearsonr(y_train,train_predict)
        residual,err,mse = plu.get_residual_stats(y_train,train_predict)

    r2 = r ** 2
    rstring = '$r^2$ = {: 5.3f}'.format(r2)
    
    #Misclassification probabilities and training set confidences
    prob_zero_train = plu.get_probabilities(class_predicted=train_predict,data_error=error_train,class_value=cv)
    train_assigned = np.zeros_like(train_predict)
    train_assigned[train_predict > cv] = 1
    misclass,misclass_mask_train = plu.find_misclassified(true_class=y_train,assigned_class=train_assigned)
    train_confidence = plu.misclass_probability(prob_zero_train,misclass_mask_train)
    
    if test_data is not None:
        prob_zero = plu.get_probabilities(class_predicted=test_predict,data_error=error_test,class_value=cv)
        test_assigned = np.zeros_like(test_predict)
        test_assigned[test_predict > cv] = 1
        misclass_test,misclass_mask_test = plu.find_misclassified(true_class=yt,assigned_class=test_assigned)
        test_confidence = plu.misclass_probability(prob_zero,misclass_mask_test)
        
    train_order = np.argsort(prob_zero_train)
    if test_data is not None:
        test_order = np.argsort(prob_zero)
    
    if group:
        boot_all = train_data
        class_predict = train_predict
        colors = train_colors
        if test_data is not None:            
            boot_all = np.concatenate((boot_all,test_data))
            class_predict = np.concatenate((class_predict,test_predict))
            colors = np.concatenate((train_colors,test_colors))
    else:
        boot_all = train_data[train_order]
        class_predict = train_predict[train_order]
        colors = train_colors[train_order]
        if test_data is not None:            
            boot_all = np.concatenate((boot_all,test_data[test_order]))
            class_predict = np.concatenate((class_predict,test_predict[test_order]))
            colors = np.concatenate((train_colors,test_colors[test_order]))
    
    #Confidence limits on the full set
    ci_upper = np.percentile(boot_all,97.5,axis=1)
    ci_lower = np.percentile(boot_all,2.5,axis=1)
    error = (ci_upper - ci_lower)/2
    
    #Split axes
    ax = axes_row[0]
    mcax = axes_row[1]
    
    #Class value plot
    
    #Axis limits for the class value plot
    ax.set_xlim(0,num_total)
    ax.set_ylim(-1,2)
    ax.set_yticks([-1,0,1,2])
    ax.set_ylabel('Predicted class, $XW^*Q^T$',size=15)
    ax.set_xlabel('Sample index (a.u.)',size=15)
    ax.set_xticks([])
    
    #Scatter plot and error bar plots for predicted class values
    plsplot = ax.scatter(np.arange(num_total),class_predict,
                     color=colors,edgecolors='k',s=30)

    plserr = ax.errorbar(np.arange(num_total),class_predict,
                         yerr=error,color='k',**errbar_dict)
    pls_class_boundaryy = ax.axhline(y=cv,**class_boundary_dict)
    
    #Misclassification probability plot
    proby_centerline = mcax.axvline(x=0.5,**class_boundary_dict)#mcax.plot((0.5,0.5),(-1.5,2.5),'k:')
    class_centerline = mcax.axhline(y=cv,**class_boundary_dict) #mcax.plot((-1,2),(cv,cv),'k:')
    
    #Misclassification probabilities
    #Training set correct classification

    mcax.scatter(
        train_confidence[~misclass_mask_train],
        train_predict[~misclass_mask_train],
        label='Correct Train',color='w',edgecolor='b')

    mcax.scatter(
        1-train_confidence[misclass_mask_train],
        train_predict[misclass_mask_train],
        label='Incorrect Train',color='w',edgecolor='r')
    if test_data is not None:
        mcax.scatter(
            test_confidence[~misclass_mask_test],
            test_predict[~misclass_mask_test],
            label='Correct Test',color='b')

        mcax.scatter(
            1-test_confidence[misclass_mask_test],
            test_predict[misclass_mask_test],
            label='Incorrect Test',color='r')
    mcax.set_xlim(-0.05,1.05)
    #mcax.set_ylim(-0.05,1.05)
    mcax.set_xticks([0,1])
    #mcax.set_yticks([0,0.5,1])
    mcax.text(0.95,0.95,rstring,ha='right',va='top',transform=mcax.transAxes)
    mcax.set_xlabel(r'$\mathsf{Pr}_\mathsf{misclass}$',size=15)
def nmr_bootstrap_annotate(ax):
    #Make the legend
    blue_dot = matplotlib.lines.Line2D([],[],color='#4444ff',marker='o',label='Control replicate',linestyle='none')
    cyan_dot = matplotlib.lines.Line2D([],[],color='c',marker='o',label='Control',linestyle='none')
    red_dot = matplotlib.lines.Line2D([],[],color='r',marker='o',label='Exposed',linestyle='none')

    leg = ax.legend(handles=[blue_dot,cyan_dot,red_dot],ncol=3,frameon=True,framealpha=1,numpoints=1)
    leg.get_frame().set_edgecolor('w')

    #Label the groups
    good_labels = np.array(Labels)[outlier_mask]

    for xpos,label in enumerate(good_labels):
        t = ax.text((xpos+0.5)*numspecs,-0.6,label,ha='center',va='center')
        p = ax.axvline(x=(xpos+0.95)*numspecs,color='k',ls='--')
def misclass_legend(mcax):
    leg = mcax.legend(scatterpoints=1,fontsize=8,framealpha=0.5,loc='lower right')
    leg.get_frame().set_edgecolor('w')

def make_grid_plot(numrows,numcols,figsize=None,plotsize=None,
                   column_width=6,row_height=4,
                   label_buffers=None,
                   ylabel_buffer=0.75,xlabel_buffer=0.5,
                   xlabel=None,ylabel=None,
                   add_buffer=False,
                   **subplots_args):

    if plotsize is not None:
        column_width,row_height = plotsize
    
    if label_buffers is not None:
        xlabel_buffer,ylabel_buffer = label_buffers
    
    full_width = numcols*column_width
    full_height = numrows*row_height
    if add_buffer:
        full_width = numcols*column_width + ylabel_buffer
        full_height = numrows*row_height + xlabel_buffer
    
    bottom_buffer = xlabel_buffer/full_height
    left_buffer = ylabel_buffer/full_width

    ylabel_pos = 0.5*(1+bottom_buffer)
    xlabel_pos = 0.5*(1+left_buffer)
  
    fs = (full_width,full_height)
    if figsize is not None:
        fs = figsize
    fig,axes = plt.subplots(numrows,numcols,figsize=fs,squeeze=False,**subplots_args)
    fig.subplots_adjust(left=left_buffer,right=1,top=1,bottom=bottom_buffer)
    
    if ylabel:
        fig.text(0,ylabel_pos,ylabel,size=15,rotation='vertical',va='center')
    if xlabel:
        fig.text(xlabel_pos,0.0,xlabel,ha="center",va="center",size=15)
        
    return fig,axes