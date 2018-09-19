import numpy as np
import sklearn.decomposition
import sklearn.cross_decomposition
import sklearn.model_selection
import sklearn.pipeline as skpipe
import math
import tqdm
import scipy

def get_probabilities(class_predicted=None,data_error=None,PLS_model=None,class_value=0.5):
    #class_predicted = PLS_model.predict(data) # Get the predicted classes for this data set
    probability_zero = [] # Empty list to be filled with the 
    for mol_num,value in enumerate(class_predicted):
        error = data_error[mol_num]
        prob_zero = 0.5 * (
            1 + math.erf(
                (class_value - value)/(error/2 * math.sqrt(2))
            )
        ) # Find where 0.5 falls on the cumulative distribution function
        probability_zero += [prob_zero]
    return np.array(probability_zero)
def class_assignment(data=None,data_error=None,PLS_model=None,class_value=0.5):
    class_predicted = data
    if PLS_model is not None:
        class_predicted = PLS_model.predict(data) # Get the predicted classes for this data set
    class_assigned = np.zeros_like(class_predicted,dtype=int)
    class_assigned[class_predicted > class_value] = 1
    return class_assigned,class_predicted
def class_assignment_boot(class_predicted_bootstrap=None,data_error=None,PLS_model=None,class_value=0.5):
    #class_predicted_bootstrap = PLS_model.predict(data)
    class_predicted = np.median(class_predicted_bootstrap,keepdims=True,axis=1)
    class_assigned = np.zeros_like(class_predicted,dtype=int)
    class_assigned[class_predicted > class_value] = 1
    return class_assigned,class_predicted,class_predicted_bootstrap
def find_misclassified(true_class=None,assigned_class=None):
    misclass = np.squeeze(abs(true_class - assigned_class.T)) # If true class == assigned class, this will be 0, otherwise 1 or -1
    misclass_mask = misclass == 1 
    return misclass,misclass_mask
def _estimate_class_boundary(predicted_class,actual_class):
    #Find which samples correspond to actual classes 1 and 0
    class_one = predicted_class[actual_class==1]
    class_zero = predicted_class[actual_class==0]
    
    #Find the mean prediction and standard deviation for all of the class 1 samples
    class_one_mean = class_one.mean()
    class_one_std = class_one.std()
    
    #Find the mean prediction and standard deviation for all of the class 0 samples
    class_zero_mean = class_zero.mean()
    class_zero_std = class_zero.std()
    
    #Compute the location where class 1 and class 0 probability density functions are equal
    log_std_ratio = np.log(class_one_std / class_zero_std)
    mean_diff = (class_one_mean - class_zero_mean) / 2
    mean_avg = (class_one_mean + class_zero_mean) / 2
    
    class_boundary = ((class_one_std) ** 2) * ((class_zero_std) ** 2) / mean_diff * log_std_ratio + mean_avg
    return class_boundary

def estimate_class_boundary(predicted_class,actual_class):
    #Find which samples correspond to actual classes 1 and 0
    class_one = predicted_class[actual_class==1]
    class_zero = predicted_class[actual_class==0]
    
    #print(class_one)
    #print(class_zero)
    
    #Find the mean prediction and standard deviation for all of the class 1 samples
    class_one_mean = class_one.mean()
    class_one_std = class_one.std()
    
    #Find the mean prediction and standard deviation for all of the class 0 samples
    class_zero_mean = class_zero.mean()
    class_zero_std = class_zero.std()
    
    #Compute the location where class 1 and class 0 probability density functions are equal
    #These are constants needed in the calculation
    log_std_ratio = np.log(class_zero_std / class_one_std)
    var_product = (class_zero_std * class_one_std) ** 2
    var_diff = class_zero_std ** 2 - class_one_std ** 2
    mean_diff_squared = (class_one_mean - class_zero_mean)**2
    
    #Coefficients in the quadratic for x
    a = var_diff
    minus_b = 2 * (class_one_mean * (class_zero_std ** 2) - class_zero_mean * (class_one_std ** 2))
    c = ((class_zero_mean * class_one_std)**2 - (class_one_mean * class_zero_std)**2) - 2*(log_std_ratio*var_product)
    
    #Simplify the discriminant
    discriminant = 2*np.sqrt(2*log_std_ratio*var_product*var_diff + var_product*mean_diff_squared)
    
    center = minus_b/(2*a)
    
    #Quadratic, so two roots
    #Quadratic, so two roots
    plus_boundary = (minus_b + discriminant)/(2*a)
    minus_boundary = (minus_b - discriminant)/(2*a)
    
    upper_boundary = max(plus_boundary,minus_boundary)
    lower_boundary = min(plus_boundary,minus_boundary)
    
    #Choose boundary based on which std is bigger (equivalent to sign of var_diff)
    class_boundary = lower_boundary 
    if upper_boundary < class_one_mean : class_boundary = upper_boundary

    return class_boundary

def bootstrap_stratified(data,classes,samples=1000,axis=0):
    num_data,num_classes = classes.shape#Get the shape of the class data
    
    full_indices = np.arange(num_data)#Get an array of the indices
    
    data_boot = []
    boot_indices = []
    for class_num in range(num_classes):
        class_mask = classes[:,class_num] == 1 #Find which samples are in this class
        this_data = data[class_mask] #Get the x data
        this_indices = full_indices[class_mask]
        #Boot the straps
        this_boot,this_boot_indices = bootstrap_data(this_data,samples=samples,axis=axis)
        #Put the data into a list
        data_boot += [this_boot]
        #use the local indices to get where these data came from in the full matrix
        boot_indices += [this_indices[this_boot_indices]]
        
    data_boot = np.concatenate(data_boot,axis=1)
    boot_indices = np.concatenate(boot_indices,axis=1)
    return data_boot,boot_indices


def bootstrap_data(data,samples=1000,axis=0):
    """Creates random indices in order to conduct bootstrap uncertainty analysis. Returns also the permuted data matrix.
    """
    array_shape = data.shape
    boot_shape = (samples,) + (array_shape[axis],)
    
    boot_indices = np.random.randint(0,array_shape[axis],boot_shape)
    
    boot_out = data[boot_indices]
    
    if len(boot_out.shape) < 3: boot_out = np.expand_dims(boot_out,axis=-1)
    
    return boot_out,boot_indices


def simple_bootstrap(xdata=None,ydata=None,
                     PLS_model=None,PLS_cv=None,PLS_bootstrap=None,sk_model=sklearn.cross_decomposition.PLSRegression,
                     cv_object=None,class_value=0.5,samples=1000,PLS_kw=None,return_boot=False):
    """Conducts a simple residual bootstrap analysis on a set of data. Computes cross-validation uncertainty.
    
    This function relies on the Y-data being bootstrapped to be one-dimensional. It also requires the model to be accept two-dimensional data. The bootstrapping is done by generating :math:`samples` random variations on the Y-data and then concatenating them into a two-dimensional array.
    
    If PLS_model is None, then PLS_cv and PLS_bootstrap are ignored. The function will create independent instances of :py:class:sk_model for each of PLS_model, PLS_cv, and PLS_boostrap.
    
    If PLS_model is not None, then it will be reused for PLS_cv and PLS_bootstrap.
    
    :key xdata: The X data used to fit the model (default None)
    :key ydata: The Y data used to fit the model (default None)
    :key PLS_model: The scikit-learn model that will be fit using X and Y
    :key PLS_cv: The scikit-learn model that will be used for cross-validation
    :key PLS_bootstrap: The scikit-learn model that will be used for bootstrapping
    :key sk_model: If PLS_model,PLS_cv,or PLS_bootstrap is None, this scikit-learn model will be used to create them
    :key cv_object: The cross-validation model that will be used for calculating cross-validation statistics
    :key class_value: The value separating the classes in PLS-DA
    :key samples: The number of samples for bootstrapping
    :key PLS_kw: The keyword arguments that will be passed to sk_model
    :key return_boot: If True, returns the PLS_bootstrap model as part of the output
    :type xdata: ndarray
    :type ydata: 1-d array
    :type PLS_model: scikit-learn model instance
    :type PLS_cv: scikit-learn model instance
    :type PLS_bootstrap: scikit-learn model instance
    :type skmodel: scikit-learn model
    :type cv_object: scikit-learn model selection instance
    :type class_value: scalar
    :type samples: int
    :type PLS_kw: dict
    :type return_boot: Boolean
    """
    
    if PLS_model is None:
        #PLS_model = sklearn.cross_decomposition.PLSRegression(**PLS_kw)
        #PLS_cv = sklearn.cross_decomposition.PLSRegression(**PLS_kw)
        #PLS_bootstrap = sklearn.cross_decomposition.PLSRegression(**PLS_kw)
        PLS_model = sk_model(**PLS_kw)
        PLS_cv = sk_model(**PLS_kw)
        PLS_bootstrap = sk_model(**PLS_kw)     
    
    print (PLS_model)
    
    if PLS_cv is None:
        PLS_cv = PLS_model
    if PLS_bootstrap is None:
        PLS_bootstrap = PLS_model
    
    tpls = PLS_model.fit(xdata,ydata)
    
    #Get class assignments for the base PLS model
    class_assigned_train,class_predicted_train = class_assignment(data=xdata,PLS_model=PLS_model,class_value=class_value)
    
    if cv_object is None:
        cv_object = sklearn.model_selection.StratifiedKFold(n_splits=6)
    
    #Cross-validate
    class_assigned_cv,class_predicted_cv = cross_validate(xdata=xdata,ydata=ydata,
                                                         PLS_model=PLS_cv,cv_object=cv_object,class_value=class_value)
    
    #Calculate residuals and mean squared errors of regression
    residual,err,mse = get_residual_stats(ydata,class_predicted_train)
    residual_cv,err_cv,msecv = get_residual_stats(ydata,class_predicted_cv)
    
    
    print('Y data shape',ydata.shape)
    print('Output shapes',class_predicted_train.shape,class_predicted_cv.shape)
    print('Meas squared error:',mse,msecv)
    
    #Calculate the pseudo degrees of freedom and the corresponding bootstrap weighting factor
    num_train = len(ydata)
    pseudo_dof = num_train * (1 - np.sqrt(mse/msecv))
    bootstrap_weight = np.sqrt(1 - pseudo_dof/num_train)
    
    #Caclulate the weighted residual vector and bootstrap it to generate the bootstrap perturbations
    residual_weighted = residual / bootstrap_weight
    residual_boot,boot_indices = bootstrap_data(residual_weighted,samples=samples)
    
    #Get the new y data generated by bootstrapping and fit the PLS model to it
    class_y_boot = np.transpose(np.squeeze(class_predicted_train) + np.squeeze(residual_boot))
    print ('predicted_train = ', class_predicted_train.shape)
    print ('residuals shape = ', residual_boot.shape)
    print ('boot data shape = ', class_y_boot.shape)
    print ('x data shape = ',xdata.shape)
    
    PLS_bootstrap.fit(xdata,class_y_boot)
    
    #Get the bootstrapped predictions from the PLS model
    class_predicted_boot = PLS_bootstrap.predict(xdata)
    
    return_out = (residual_cv,err_cv,msecv,class_predicted_boot,class_predicted_train,)
    
    if return_boot:
        return_out += (PLS_bootstrap,)
    
    return return_out#residual_cv,err_cv,msecv,class_predicted_boot,class_predicted_train

def bootstrap(xdata=None,ydata=None,validdata=None,
                     PLS_model=None,PLS_cv=None,PLS_bootstrap=None,sk_model=sklearn.cross_decomposition.PLSRegression,regression=False,
                     cv_object=None,class_value=0.5,samples=1000,PLS_kw=None,return_scores=False,return_loadings=False,tq=True):
    """Conducts a simple residual bootstrap analysis on a set of data. Computes cross-validation uncertainty.
    
    This function performs a full bootstrap and makes no assumption about the shape or structure of the Y data. Each bootstrap sample will have an independent model fit to it.
    
    If PLS_model is None, then PLS_cv and PLS_bootstrap are ignored. The function will create independent instances of :py:class:sk_model for each of PLS_model, PLS_cv, and PLS_boostrap.
    
    If PLS_model is not None, then it will be reused for PLS_cv and PLS_bootstrap.
    
    :key xdata: The X data used to fit the model (default None)
    :key ydata: The Y data used to fit the model (default None)
    :key validdata: Additional data not used to fit the model but for which uncertainty will be calculated
    :key PLS_model: The scikit-learn model that will be fit using X and Y. If None, a new model will be created from sk_model
    :key PLS_cv: The scikit-learn model that will be used for cross-validation. If None, same as PLS_model.
    :key PLS_bootstrap: The scikit-learn model that will be used for bootstrapping. If None, same as PLS_model.
    :key sk_model: If PLS_model,PLS_cv,or PLS_bootstrap is None, this scikit-learn model will be used to create them
    :key cv_object: The cross-validation model that will be used for calculating cross-validation statistics
    :key class_value: The value separating the classes in PLS-DA
    :key samples: The number of samples for bootstrapping
    :key PLS_kw: The keyword arguments that will be passed to sk_model
    :key return_scores: If True, returns the scores of the PLS_bootstrap model as part of the output
    :key return_loadings: If True, returns the loadings of the PLS_bootstrap model as part of the output
    :type xdata: ndarray
    :type ydata: 1-d array
    :type PLS_model: scikit-learn model instance
    :type PLS_cv: scikit-learn model instance
    :type PLS_bootstrap: scikit-learn model instance
    :type s_model: scikit-learn model
    :type cv_object: scikit-learn model selection instance
    :type class_value: scalar
    :type samples: int
    :type PLS_kw: dict
    :type return_scores: Boolean
    :type return_loadings: Boolean
    """
    if PLS_model is None:
        #PLS_model = sklearn.cross_decomposition.PLSRegression(**PLS_kw)
        #PLS_cv = sklearn.cross_decomposition.PLSRegression(**PLS_kw)
        #PLS_bootstrap = sklearn.cross_decomposition.PLSRegression(**PLS_kw)
        PLS_model = sk_model(**PLS_kw)
        PLS_cv = sk_model(**PLS_kw)
        PLS_bootstrap = sk_model(**PLS_kw)     
    
    if PLS_cv is None:
        PLS_cv = PLS_model
    if PLS_bootstrap is None:
        PLS_bootstrap = PLS_model
    
    if cv_object is None:
        cv_object = sklearn.model_selection.StratifiedKFold(n_splits=6)
    
    tpls = PLS_model.fit(xdata,ydata)
    scores_base = None
    try:
        scores_base = PLS_model.transform(xdata)
    except AttributeError: #Not all classification models have a transform attribute, so 
        print('No transform function, skipping scores')
    
    #Get class assignments for the base PLS model (do not do this for regression models)
    class_assigned_train,class_predicted_train = class_assignment(data=xdata,PLS_model=PLS_model,class_value=class_value)
    
    #Cross-validate
    class_assigned_cv,class_predicted_cv = cross_validate(xdata=xdata,ydata=ydata,
                                                         PLS_model=PLS_cv,cv_object=cv_object,class_value=class_value)
    
    #Calculate residuals and mean squared errors of regression
    residual,err,mse = get_residual_stats(ydata,class_predicted_train.T)
    residual_cv,err_cv,msecv = get_residual_stats(ydata,class_predicted_cv.T)
    
    #Calculate the pseudo degrees of freedom and the corresponding bootstrap weighting factor
    num_train = len(ydata)
    pseudo_dof = num_train * (1 - np.sqrt(mse/msecv))
    bootstrap_weight = np.sqrt(1 - pseudo_dof/num_train)
    
    #Caclulate the weighted residual vector and bootstrap it to generate the bootstrap perturbations
    residual_weighted = residual / bootstrap_weight
    residual_boot,boot_indices = bootstrap_data(residual_weighted,samples=samples)
    
    #print residual_boot.shape
    
    
    #class_predicted_boot = np.empty((len(ydata),0,))
    #if validdata is not None:
    #    class_valid_boot = np.empty((len(validdata),0,))
    #scores_boot = np.empty((0,PLS_kw['n_components']))
    #scores_valid = np.empty((0,PLS_kw['n_components']))
    #load_boot = np.empty((xdata.shape[1],0,))
    
    class_predicted_boot = []
    if validdata is not None:
        class_valid_boot = []
    scores_boot = []
    scores_valid = []
    load_boot = []
    
    if tq:
        iterate = tqdm.tqdm_notebook(residual_boot)
    else:
        iterate = residual_boot
            
    
    
    for bt in iterate:
        #print (class_predicted_train.shape)
        #print (bt.shape)
        if len(class_predicted_train.shape) > 1: #Check to make sure the shapes of class_predicted_train and bt align properly
            class_y_boot = class_predicted_train + bt
        else:
            class_y_boot = class_predicted_train + np.squeeze(bt)
        #print (class_y_boot.shape)
        
        #print bt.shape
        #print class_y_boot.shape
        #print class_predicted_train.shape
        
        #raise ValueError
        
        PLS_bootstrap.fit(xdata,class_y_boot)
        
        cp_boot = PLS_bootstrap.predict(xdata)
        #class_predicted_boot = np.concatenate((class_predicted_boot,cp_boot),axis=1)
        array_dims = len(cp_boot.shape)
        if array_dims > 1: #the model gives a 2-D array
            class_predicted_boot += [cp_boot]
        else: #the model gives a 1-D array and we have to add a new singleton axis to it
            class_predicted_boot += [cp_boot[:,np.newaxis]]
        
        try:
            sc_star = PLS_bootstrap.transform(xdata)
            procrustes_rotation,procrustes_scale = scipy.linalg.orthogonal_procrustes(sc_star,scores_base) 
        
        
            sc_boot = np.dot(sc_star,procrustes_rotation)
            #scores_boot = np.concatenate((scores_boot,sc_boot),axis=0)
            scores_boot += [sc_boot]
            ld_star = PLS_bootstrap.x_loadings_
            ld_boot =  np.dot(ld_star,procrustes_rotation)        
            #load_boot = np.concatenate((load_boot,ld_boot),axis=1)
            load_boot += [ld_boot]
        except AttributeError:
            pass
        
        
        
        if validdata is not None:
            valid_boot = PLS_bootstrap.predict(validdata)
            #class_valid_boot = np.concatenate((class_valid_boot,valid_boot),axis=1)
            if array_dims > 1:
                class_valid_boot += [valid_boot]
            else:
                class_valid_boot += [valid_boot[:,np.newaxis]]
            try:
                sc_v = PLS_bootstrap.transform(validdata)
                #scores_valid = np.concatenate((scores_valid,sc_v),axis=0)
                scores_valid += [sc_v]
            except AttributeError:
                pass
        
        
    #print(class_predicted_boot[0].shape)
    class_predicted_boot =  np.concatenate(class_predicted_boot,axis=1)
    
    
    return_out = (residual_cv,err_cv,msecv,class_predicted_boot,class_predicted_train,)
    
    if validdata is not None:
        class_valid_boot = np.concatenate(class_valid_boot,axis=1)
        return_out += (class_valid_boot,)
    
    if return_scores:
        scores_boot = np.concatenate(scores_boot,axis=0)
        return_out += (scores_boot,)
        if validdata is not None:
            scores_valid = np.concatenate(scores_valid,axis=0)
            return_out += (scores_valid,)
    if return_loadings:
        load_boot = np.concatenate(load_boot,axis=1)
        return_out += (load_boot,)
    
    return return_out

def bootstrap_unc(xdata=None,ydata=None,valid_data=None,
                        cv_object=None,
                        samples=1000,class_value=0.5,
                        PLS_kw=None,return_scores=False,tq=True):
    """Computes the uncertainty in a bootstrap analysis by leave-one-out cross-validation.
    
    For each sample, the uncertainty is calculated by fitting the other samples to the model, calculating the bootstrap uncertainty and then calculating the uncertainty in the held-out sample.
    
    :key xdata: The X data used to fit the model (default None)
    :key ydata: The Y data used to fit the model (default None)
    :key valid_data: Additional data not used to fit the model but for which uncertainty will be calculated
    :key cv_object: The cross-validation model that will be used for calculating cross-validation statistics
    :key samples: The number of samples for bootstrapping
    :key class_value: The value separating the classes in PLS-DA
    :key PLS_kw: The keyword arguments that will be passed to sk_model
    :key return_scores: If True, returns the scores of the PLS_bootstrap model as part of the output
    :type xdata: ndarray
    :type ydata: ndarray
    :type cv_object: scikit-learn model selection instance
    :type class_value: scalar
    :type samples: int
    :type PLS_kw: dict
    :type return_scores: Boolean
    """
    
    pls_comps = PLS_kw['n_components']
    
    ci_boot = np.empty((0,2))
    predict_boot = np.empty((0,samples))
    scores_full = np.empty((0,pls_comps))
    
    ci_boot = []
    predict_boot = []
    scores_full = []
    
    #This should be the base PLS model
    base_pls = sklearn.cross_decomposition.PLSRegression(**PLS_kw)
    _ = base_pls.fit(xdata,ydata)
    class_predicted_train = base_pls.predict(xdata)
    
    if tq:
        iterate = tqdm.tqdm_notebook(cv_object.split(xdata,ydata),total=len(ydata))
    else:
        iterate = cv_object.split(xdata,ydata)
    
    #for train_index,test_index in tqdm.tqdm_notebook(cv_object.split(xdata,ydata),total=len(ydata)):
    for train_index,test_index in iterate:
        
        # This is the element that is being removed and having its uncertainty calculated
        this_x = xdata[test_index]
        this_y = ydata[test_index]
        
        #This is everything else
        not_this_x = xdata[train_index]
        not_this_y = ydata[train_index]
        
        #Create the local PLS models and the stratified K fold cross validation object
        this_pls = sklearn.cross_decomposition.PLSRegression(**PLS_kw)
        this_pls_cv = sklearn.cross_decomposition.PLSRegression(**PLS_kw)
        this_pls_boot = sklearn.cross_decomposition.PLSRegression(**PLS_kw)
        this_cv = sklearn.model_selection.StratifiedKFold(n_splits=6)
        
        
        #Boot the straps
        boot_out = bootstrap(xdata=not_this_x,
                                   ydata=not_this_y,
                                   validdata=this_x,
                                   PLS_model=this_pls,
                                   PLS_cv = this_pls_cv,
                                   PLS_bootstrap=this_pls_boot,
                                   PLS_kw=PLS_kw,
                                   cv_object=this_cv,
                                   samples=samples,
                                   class_value=class_value,
                                   return_scores=return_scores,
                                   tq=tq)
        
        #class_predicted_boot is the bootstrap predictions for not_this_x
        #class_y_base_predict is the single-model class prediction for not_this_x
        #class_y_boot_predict is the bootstrap prediction for this_x
        
        #rcv,ecv,msecv,class_predicted_boot,class_y_base_predict,class_y_boot_predict,scores_boot,scores_valid = boot_out
        rcv,ecv,msecv,boot_ydata,train_ydata,boot_this_x,scores_boot,scores_valid=boot_out
        
        #rcv,ecv,msecv,class_predicted_boot,class_predicted_train,scores_boot = lbt_out
        
        #return_out = (residual_cv,err_cv,msecv,class_predicted_boot,class_predicted_train,class_valid_boot,scores_boot,)
        
        
        #Compute the confidence interval
        train_ydata = np.median(boot_this_x,axis=1)
        ci = np.percentile(boot_this_x,[2.5,97.5],axis=1)
        ci = np.array([ci])
        ci = ci - train_ydata + class_predicted_train[test_index]
        #ci_boot = np.concatenate((ci_boot,ci),axis=0)
        ci_boot += [ci]
        
        #rtrt = (ci,class_y_base_predict,class_y_boot_predict,scores_boot)
        #ci,this_predict,class_boot,scores = rtrt
        

        class_y_boot_predict = boot_this_x - train_ydata + class_predicted_train[test_index] 
        
        #predict_boot = np.concatenate((predict_boot,class_y_boot_predict),axis=0)
        predict_boot += [class_y_boot_predict]
        
        #scores_full = np.concatenate((scores_full,scores_valid),axis=0)
        scores_full += [scores_valid]
    
    ci_boot = np.concatenate(ci_boot,axis=0)
    predict_boot = np.concatenate(predict_boot,axis=0)
    scores_full = np.concatenate(scores_full,axis=0)
    
    return ci_boot,predict_boot,scores_full


def pca_bootstrap(xdata=None,ydata=None,groups=None,validdata=None,
                  PCA_model=None,PCA_cv=None,PCA_bootstrap=None,skmodel=sklearn.decomposition.PCA,scaler=None,
                  cv_object=None,samples=1000,PCA_kw=None,tq=True):
    """Conducts a residual bootstrap analysis on a set of data. Computes cross-validation uncertainty.
    
    This function is the same as :py:func:`bootstrap` but works for unsupervised models such as PCA
    
    If PLS_model is None, then PLS_cv and PLS_bootstrap are ignored. The function will create independent instances of :py:class:sk_model for each of PLS_model, PLS_cv, and PLS_boostrap.
    
    If PLS_model is not None, then it will be reused for PLS_cv and PLS_bootstrap.
    
    :key xdata: The X data used to fit the model (default None)
    :key ydata: The Y data used to fit the model (default None)
    :key PCA_model: The scikit-learn model that will be fit using X
    :key PCA_cv: The scikit-learn model that will be used for cross-validation
    :key PCA_bootstrap: The scikit-learn model that will be used for bootstrapping
    :key sk_model: If PLS_model,PLS_cv,or PLS_bootstrap is None, this scikit-learn model will be used to create them
    :key scaler: The scikit-learn preprocessing object used to preprocess the data. This will be put into a 
    :key cv_object: The cross-validation model that will be used for calculating cross-validation statistics
    :key samples: The number of samples for bootstrapping
    :key PCA_kw: The keyword arguments that will be passed to sk_model
    :key return_boot: If True, returns the PLS_bootstrap model as part of the output
    :type xdata: ndarray
    :type ydata: ndarray
    :type PCA_model: scikit-learn estimator instance
    :type PCA_cv: scikit-learn estimator instance
    :type PCA_bootstrap: scikit-learn estimator instance
    :type sk_model: scikit-learn estimator class
    :type sk_model: scikit-learn preprocessing instance
    :type cv_object: scikit-learn model selection instance
    :type class_value: scalar
    :type samples: int
    :type PCA_kw: dict
    :type return_boot: Boolean
    """
    estimator_step_name = 'PCA'
    scaler_step_name = 'scaler'
    
    if PCA_model is None:
        PCA_model = skmodel(**PCA_kw)
        PCA_cv = skmodel(**PCA_kw)
        PCA_bootstrap = skmodel(**PCA_kw)
    
    steps_model = [(estimator_step_name,PCA_model)]
    steps_cv = [(estimator_step_name,PCA_cv)]
    steps_bootstrap = [(estimator_step_name,PCA_bootstrap)]
    
    if scaler is not None:
        steps_model = [(scaler_step_name, scaler),(estimator_step_name,PCA_model)]
        steps_cv = [(scaler_step_name, scaler),(estimator_step_name,PCA_cv)]
        steps_bootstrap = [(scaler_step_name, scaler),(estimator_step_name,PCA_bootstrap)]
    
    pipe_model = skpipe.Pipeline(steps_model)
    pipe_cv = skpipe.Pipeline(steps_cv)
    pipe_bootstrap = skpipe.Pipeline(steps_bootstrap)
    
    tpca = pipe_model.fit(xdata)
    scores = pipe_model.transform(xdata)
    x_trans = pipe_model.inverse_transform(scores)
    
    #cross_validation
    x_cv = pca_cross_validate(xdata=xdata,ydata=ydata,groups=groups,PCA_model=pipe_cv,cv_object=cv_object,PCA_kw=PCA_kw)
    
    residual,err,mse = get_residual_stats(xdata,x_trans)
    residual_cv,err_cv,msecv = get_residual_stats(xdata,x_cv)    
    
    #print mse,msecv
    
    num_train = len(ydata)
    pseudo_dof = num_train * (1 - np.sqrt(mse/msecv))
    bootstrap_weight = np.sqrt(1 - pseudo_dof/num_train)    
    
    #print bootstrap_weight
    
    residual_weighted = residual / bootstrap_weight
    #x_boot,boot_indices = pls_uncertainty.bootstrap_data(residual_weighted,samples=samples)
    x_boot,boot_indices = bootstrap_data(xdata,samples=samples)
    #x_boot,boot_indices = bootstrap_stratified(xdata,classes=ydata,samples=samples)
    #print boot_indices.shape
    #print boot_indices
    
    #print x_boot.shape    
    #x_boot = x_boot * 2
    
    if tq:
        iterate = tqdm.tqdm_notebook(zip(x_boot,boot_indices))
    else:
        iterate = zip(x_boot,boot_indices)
    
    scores_boot = np.empty((0,PCA_kw['n_components']))
    class_boot = np.empty((0,ydata.shape[1]))
    comps_boot = np.empty((0,xdata.shape[1]))
    
    scores_boot = []
    class_boot = []
    comps_boot = []
    
    components_base = pipe_model.named_steps[estimator_step_name].components_.T
    base_scores_boot = pipe_model.transform(xdata) #Get the scores for the base X for the base model

    
    for bt,index in iterate:
        #print bt.shape
        #print xdata.shape
        #print scores.shape
        #break
        
        x_bt = bt# xdata + bt
        y_bt = ydata#[index]
        
        #sc_boot = PCA_model.transform(x_bt)
        
        bpca = pipe_bootstrap.fit(x_bt) #Fit the model to the current bootstrap X
        
        #scores_boot_this = PCA_bootstrap.transform(x_bt) #Get the scores for the current bootstrap X for the bootstrap model
        #base_scores_boot = PCA_model.transform(x_bt) #Get the scores for the current bootstrap X for the base model
        
        scores_boot_this = pipe_bootstrap.transform(xdata) #Get the scores for the base X for the bootstrap model
        
        #print base_scores_boot.shape
        #print components_base.shape
        
        # This is the scores + components matrix that Procrustes must match
        procrustes_target = np.concatenate((base_scores_boot,components_base),axis=0)
        # This is the matrix that Procrustes will operate on
        procrustes_operand = np.concatenate((scores_boot_this,pipe_bootstrap.named_steps[estimator_step_name].components_.T),axis=0)
        
        #Procrustes the bootstrap scores matrix so that our bootstrap scores match the scores of the base model
        #procrustes_rotation,procrustes_scale = scipy.linalg.orthogonal_procrustes(scores_boot_this,base_scores_boot) 
        procrustes_rotation,procrustes_scale = scipy.linalg.orthogonal_procrustes(procrustes_operand,procrustes_target) 
        
        #Compute the Procrustesed scores and use that as the bootstrap scores result
        sc_boot = np.dot(scores_boot_this,procrustes_rotation)
        cp_boot = np.dot(pipe_bootstrap.named_steps[estimator_step_name].components_.T,procrustes_rotation)
        #print comps_boot.shape
        #print cp_boot.shape
        
        #if PCA_bootstrap.components_[0,0] > 0: sc_boot[0] = sc_boot[0] * -1
        
        scores_boot += [sc_boot]
        class_boot += [y_bt]
        comps_boot += [cp_boot.T]
        
    scores_boot = np.concatenate((scores_boot),axis=0)
    class_boot = np.concatenate((class_boot),axis=0)
    comps_boot = np.concatenate((comps_boot),axis=0)
    
    return scores,scores_boot,class_boot,comps_boot

def pca_cross_validate(xdata=None,ydata=None,groups=None,PCA_model=None,cv_object=None,PCA_kw=None):
    """Conducts a cross-validation analysis on a set of data using an unsupervised classification algorithm
    """
    if PCA_model is None:
        PCA_model = sklearn.decomposition.PCA(**PCA_kw)
    
    num_features = xdata.shape[1]
    
    #Create empty arrays for the cross_validation results
    x_transcv = np.transpose(np.array([[]] * num_features))
    
    if groups is None:
        iterate = cv_object.split(xdata,ydata)
    else:
        iterate = cv_object.split(xdata,ydata,groups)
    
    for train_index,test_index in iterate:
        #Break out the training and test sets for cross-validation
        x_train,x_test = xdata[train_index],xdata[test_index]
        
        #Calibrate the PLS model against the CV training set and get the class assignments for the test set
        PCA_model.fit(x_train)
        scores = PCA_model.transform(x_test)
        x_trans = PCA_model.inverse_transform(scores)
        
        #print x_transcv.shape,x_trans.shape
        
        #print class_predicted_cv.shape
        #print this_y.shape
        x_transcv = np.concatenate((x_transcv,x_trans))
    return x_transcv

def _cross_validate(xdata=None,ydata=None,PLS_model=None,cv_object=None,
                   sk_model=sklearn.cross_decomposition.PLSRegression,PLS_kw=None,class_value=0.5):
    """Conducts a cross-validation analysis on a set of data using a regression algorithm
    
    :param xdata:
    :param ydata:
    :param PLS_model:
    :param cv_object:
    :param PLS_kw:
    :param class_value:
    
    """
    if PLS_model is None:
        PLS_model = sk_model(**PLS_kw)
    
    
    try:
        #The number of classes is equal to the number of class values supplied
        num_classes = len(class_value)
    except TypeError:
        #If the class value is a float, it won't have a len(), so len(class_value) returns TypeError. If we get TypeError, assume the user simply gave a single float for class_value and meant for there to be one class
        num_classes = 1
    
    #Create empty arrays for the cross_validation results
    #class_predicted_cv = np.transpose(np.array([[]] * num_classes))
    #class_assigned_cv = np.transpose(np.array([[]] * num_classes))
    class_predicted_cv = []
    class_assigned_cv = []
    
    for train_index,test_index in cv_object.split(xdata,ydata):
        #Break out the training and test sets for cross-validation
        x_train,x_test = xdata[train_index],xdata[test_index]
        y_train,y_test = ydata[train_index],ydata[test_index]
        
        #Calibrate the PLS model against the CV training set and get the class assignments for the test set
        PLS_model.fit(x_train,y_train)
        this_assigned,this_y = class_assignment(data=x_test,PLS_model=PLS_model,class_value=class_value)
        
        #print class_predicted_cv.shape
        #print this_y.shape
        #class_predicted_cv = np.concatenate((class_predicted_cv,this_y))
        #class_assigned_cv = np.concatenate((class_assigned_cv,this_assigned))
        #array_dims = len(this_y.shape)
        #print(array_dims)
        #if array_dims < 2:
        #    this_y = this_y[:,np.newaxis]
        #    this_assigned = this_assigned[:,np.newaxis]        
        class_predicted_cv += [this_y]
        class_assigned_cv += [this_assigned]
    class_predicted_cv = np.concatenate(class_predicted_cv)
    class_assigned_cv = np.concatenate(class_assigned_cv)
    
    return class_assigned_cv,class_predicted_cv

def cross_validate(xdata=None,ydata=None,PLS_model=None,cv_object=None,
                   sk_model=sklearn.cross_decomposition.PLSRegression,PLS_kw=None,class_value=0.5):
    """Conducts a cross-validation analysis on a set of data using a regression algorithm.
    
    This function is essentially a pass-through to :py:class:`sklearn.model_selection.cross_val_predict`, and then does PLS-DA class assignments
    
    :key xdata: The X data used to fit the model (default None)
    :key ydata: The Y data used to fit the model (default None)
    :key PLS_model: The scikit-learn model that will be fit using X and Y
    :key cv_object: The cross-validation model that will be used for calculating cross-validation statistics
    :key sk_model: If PLS_model,PLS_cv,or PLS_bootstrap is None, this scikit-learn model will be used to create them
    :key PLS_kw: The keyword arguments that will be passed to sk_model
    :key class_value: The value separating the classes in PLS-DA
    :type xdata: ndarray
    :type ydata: ndarray
    :type PLS_model: scikit-learn model instance
    :type sk_model: scikit-learn model
    :type cv_object: scikit-learn model selection instance
    :type class_value: scalar
    :returns: class_assigned_cv, which is the dummy variable array in PLS-DA, and class_predicted_cv, the array of PLS predictions
    """
    if PLS_model is None:
        PLS_model = sk_model(**PLS_kw)
    
    
    try:
        #The number of classes is equal to the number of class values supplied
        num_classes = len(class_value)
    except TypeError:
        #If the class value is a float, it won't have a len(), so len(class_value) returns TypeError. If we get TypeError, assume the user simply gave a single float for class_value and meant for there to be one class
        num_classes = 1
    
    y_cv = sklearn.model_selection.cross_val_predict(PLS_model,xdata,ydata,cv=cv_object)
    
    class_assigned_cv,class_predicted_cv = class_assignment(data=y_cv,class_value=class_value)
    
    return class_assigned_cv,class_predicted_cv

def get_residual_stats(true_y,predicted_y):
    """Calculates the residual, the squared error, and mean squared error given a true y and model-estimated y
    """
    residual = np.squeeze(predicted_y) - np.squeeze(true_y) # Use numpy squeeze to avoid length-1 mismatches
    if len(residual.shape) < 2: np.expand_dims(residual,axis=-1)
    
    err = residual ** 2
    
    mse = err.sum() / len(true_y)
    return residual,err,mse

def _bootstrap_uncertainty(element_number,
                          xdata=None,ydata=None,valid_data=None,
                          samples=1000,class_value=0.5,
                          PLS_kw=None,return_scores=False):
    """Conduct a residual bootstrap uncertainty analysis based on leave-one-out cross-validation (leave one sample out, conduct bootstrap on the remaining samples, and use the predictions of the left-out sample as a measure of uncertainty)
    """
    #Create the array mask to remove one element from the model array
    # this is the element whose uncertainty we are calculating
    mask=np.ones(ydata.shape[0],dtype=bool)
    mask[element_number] = False
    
    #print 
    
    #Pop out the X and Y values for this element
    # This is the data set against which we will do PLS and bootstrap
    not_this_x = xdata[mask]
    not_this_y = ydata[mask]
    
    # This is the element whose uncertainty we are calculating using PLS and bootstrap
    this_x = xdata[~mask]
    this_y = ydata[~mask]
    
    #Create the PLS and cross-validation models
    this_pls = sklearn.cross_decomposition.PLSRegression(**PLS_kw)
    this_pls_cv = sklearn.cross_decomposition.PLSRegression(**PLS_kw)
    this_pls_boot = sklearn.cross_decomposition.PLSRegression(**PLS_kw)
    this_cv = sklearn.model_selection.KFold(n_splits=6)
    
    #Fit the base PLS model, leaving out the element of interest
    this_pls.fit(not_this_x,not_this_y)
    
    boot_out = simple_bootstrap(xdata=not_this_x,
                                ydata=not_this_y,
                                PLS_model=this_pls,
                                PLS_cv = this_pls_cv,
                                PLS_bootstrap=this_pls_boot,
                                cv_object=this_cv,
                                samples=samples,
                                class_value=class_value)
    
    residual_cv,err_cv,msecv,class_predicted_boot,class_predicted_train = boot_out
    
    #Generate predictions for the element whose uncertainty we want
    class_y_boot_predict = this_pls_boot.predict(this_x)
    class_y_base_predict = this_pls.predict(this_x)
    x_score_this = this_pls_boot.transform(xdata)
    class_y_validation = None

    
    #Get 95% confidence intervals for this element
    ci = np.percentile(class_y_boot_predict,[2.5,97.5])
    ci = np.array([ci])
    class_y_base_predict = np.median(class_y_boot_predict)
    
    return_out = (ci,class_y_base_predict,class_y_boot_predict,)
    
    if valid_data is not None:
        class_y_validation = this_pls_boot.predict(valid_data)
        return_out += (class_y_validation,)
        #return ci,class_y_base_predict,class_y_boot_predict,class_y_validation
    if return_scores:
        return_out += (x_score_this,)
    return return_out
    #return ci,class_y_base_predict,class_y_boot_predict

def misclass_probability(probability_zero,misclass_mask):
    """Estimate the misclassification probability of a sample, which is based on the confidence level of the prediction compared to the true value.
    """
    misclass_prob = np.zeros_like(probability_zero)
    for element,(p_zero,misclass) in enumerate(zip(probability_zero,misclass_mask)):
        if p_zero > 0.5:
            misclass_prob[element] = 1 - p_zero
        else:
            misclass_prob[element] = p_zero
        #if misclass:
        #    misclass_prob[element] = 1 - misclass_prob[element]
    return misclass_prob