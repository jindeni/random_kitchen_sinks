#!/usr/local/bin/python
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------
#
# Purpose: this code plays around with the random kitchen sink model described in the following paper:
#
#                    A. Rahimi and B. Recht, "Weighted Sums of Random Kitchen Sinks: Replacing Minimization With Randomization In Learning"
#
#          using a dataset of signal and background data generated from two bivariate normal distribution located (-1.0,-1.0) and (+1.0,+1.0),
#          respectively, each with a standard deviation of 2.0.
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------

# load the system packages

#import sys

# load the two computing packages

import numpy as np
import scipy.optimize as optimizer

# load package to make sample datasets

from   sklearn.datasets import make_gaussian_quantiles   # used to generate a 2d gaussian distribution

# load plotting packages

import matplotlib
import matplotlib.cm     as cm
import matplotlib.mlab   as mlab
import matplotlib.pyplot as plt

# load scikit-learn packages for BDT ... used to compare against random kitchen sink results

from sklearn.ensemble    import AdaBoostClassifier
from sklearn.tree        import DecisionTreeClassifier

# set numpy printf format for floats so that a printed vector looks nicer

np.set_printoptions( formatter = { 'float' : '{: 0.3f}'.format } ) # set default numpy print format to %.3f for floats

#-----------------------------------------------------------------------------------------------------------------------------------------------------
#
# generate_dataset( ) : generates a dataset where : background - 2d normal position at (  1.0 ,  1.0 ) with variance of 4.0, i.e., std_dev = 2.0
#                                                   signal     - 2d normal position at ( -1.0 , -1.0 ) with variance of 4.0, i.e., std_dev = 2.0
#
#                       the returned dataset consists of the following objects:
#
#                       number_of_returns - number of points in the sample (background + signal)
#                       sample_indices    - a list of indices to each sample (useful for 'for ... in ...' loops
#                       sample            - the actual sample (numpy array of 'n' 2d dimensional vectors
#                       sample_x          - the x-component of each sample in the sample array (for convenience of plotting sample data)
#                       sample_y          - the x-component of each sample in the sample array (for convenience of plotting sample data)
#                       sample_classid    - the classid assigned to each sample (=1 for signal or =-1 for background)
#                       sample_x_min      - minimum x value for the sample
#                       sample_x_max      - maximum x value for the sample
#                       sample_y_min      - minimum y value for the sample
#                       sample_y_max      - maximum y value for the sample
#                       sample_min        - minimum x or y value for the sample (for making symmetric xy plots)
#                       sample_max        - maximum x or y value for the sample (for making symmetric xy plots)
#
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def generate_dataset( number_of_background_points , number_of_signal_points ) :

    debug_routine = False # set =True to have debug output generated
    
    number_of_points = number_of_background_points + number_of_signal_points

    sample_indices   = np.arange( 0 , number_of_points , 1 )

    ( signal_points , signal_classid )         = make_gaussian_quantiles( mean=(1.0,1.0)   , cov=4.0 , n_classes=1 , n_samples=number_of_signal_points     )
    signal_classid = np.empty( number_of_signal_points ) ; signal_classid.fill(  1.0 )

    ( background_points , background_classid ) = make_gaussian_quantiles( mean=(-1.0,-1.0) , cov=4.0 , n_classes=1 , n_samples=number_of_background_points )
    background_classid = np.empty( number_of_background_points ) ; background_classid.fill( -1.0 )

    sample          = np.concatenate( ( signal_points , background_points ) , axis = 0 )   # stack by rows, not column
    sample_x        = sample[ : , 0 ]                                                      # x -> coordinate 0 [for convenience when debugging]
    sample_y        = sample[ : , 1 ]                                                      # y -> coordinate 1 [for convenience when debugging]
    sample_classid  = np.concatenate( ( signal_classid , background_classid ) , axis = 0 ) # repeat sample stack on classid

    if ( debug_routine ) :
       for II in sample_indices :
           print "sample : sample[%2d] = ( %9.6f , %9.6f ) - sample_classid = %d [validation]" % ( II , sample[II,0] , sample[II,1] , sample_classid[II] )

    sample_x_max = np.amax( sample_x ) ; sample_x_min = np.amin( sample_x )
    sample_y_max = np.amax( sample_y ) ; sample_y_min = np.amin( sample_y )
    sample_max   = max( sample_x_max , sample_y_max )
    sample_min   = min( sample_x_min , sample_y_min )

    if ( debug_routine ) :
       print "max info : x_max = %9.6f , y_max = %9.6f -> max = %9.6f [validation]" % ( sample_x_max , sample_y_max , sample_max )
       print "min info : x_min = %9.6f , y_min = %9.6f -> min = %9.6f [validation]" % ( sample_x_min , sample_y_min , sample_min )

    return( number_of_points , sample_indices , sample , sample_x , sample_y , sample_classid , \
            sample_x_min , sample_x_max , sample_y_min , sample_y_max , sample_min , sample_max )

#---------------------------------------------------------------------------------------------------------------------------------------------------
#
# precompute_decision_stumps_for_training_sample( ) : computes the values of the data stumps for each element in the training sample.  During the
#                                                     optimization where the alpha coefficients are computed, the data stumps do not change.  So,
#                                                     it is best to do this operation once and keep the result in memory.  The memory usage is
#                                                     equal to the number_of_training_samples * number_of_features which should be fine for my
#                                                     purposes.
#
# The input arguments are:
#
# (1) number_of_training_points - the number of points in the training sample (could eliminate with sample.size() operation)
# (2) sample                    - the training sample, format: sample[ : , 0 ] - x value / sample [ : , 1 ] = y value
# (3) number_of_features        - the number of decision stumps
# (4) omega_components          - the axis on which each decision stump acts (here, either =0 for x-axis for =1 for y-axis) - array[number_of_features]
# (5) omega_thresholds_to_use   - the thresholds to apply in each decision stump - array[number_of_features]
# 
# and the returned output is:
#
# (1) decision_stumps_for_training_sample - an array of size [ sample number , feature number ] containing the +1 or -1 values for each decision
#                                           stump when applied to each training sample
#
#---------------------------------------------------------------------------------------------------------------------------------------------------

def generate_decision_stumps_for_training_sample( number_of_training_points , sample ,
                                                  number_of_features , omega_components , omega_thresholds_to_use ) :

    decision_stumps_for_training_samples = np.zeros( shape=(number_of_training_points,number_of_features) ) # [ sample number , feature number ]
    for KK in feature_indices :
        for II in sample_indices :
            decision_stumps_for_training_samples[II,KK] = np.sign( sample[II,omega_components[KK]] - omega_thresholds_to_use[KK] )
            
    return decision_stumps_for_training_samples # remember: the array subscript ordering is [ sample number , feature number ]

#---------------------------------------------------------------------------------------------------------------------------------------------------
#
# compute_f_for_training_sample ( ) - computes the model function for the training sample.
#
# This function is mostly around for debugging purposes as the compute_loss_function_for_training_sample( ) is the one used by the optimizer since
# that function should only return the actual loss function value.  I kept this routine around as it was just nice to have the in-between values in
# the computation available as I debugged the code.
#
# Oh, yeah, for the record, the function being computed is:
#
#                          sum_over_all_training_samples_m of [ sum_over_all_features_k of ( alpha_k * decision_stump_k_m ) ]
#
# where the decision_stump_k_m, the k-th feature function (i.e., decision stump) value for the m-th training sample, is either +1 or -1.  The
# alpha_k coefficiencts are not constrained per se, but are effectively constrained during the optimization fit to the training the datasets via
# regulization.  In practice, the resulting fit coefficients should be normalized after the fit so the fitted model function generates a result
# ranging from -1 to 1 as that makes defining a cut on that function easier.  I didn't bother to do that as I am only playing with the random
# kitchen sink idea at the moment.
#
# The input arguments are:
#
# (1) alpha_coefficients                  - the weights on the decision stump feature functions - array[number_of_features]
# (2) decision_stumps_for_training_sample - pre-computed results of the decision stumps applied to each training sample - array[samples,number_of_features]
# (3) sample_classid                      - the 'true' class id for each training sample - array[number_of_samples]
# 
# and the returned output is:
#
# (1) f_terms_for_training_sample             - the f-values for each sample in the training sample - array[number_of_samples]
# (2) f_value_for_training_sample             - the sum of the f-values for the whole training sample (useful?)
# (3) model_minus_truth_for_training_sample   - the difference between the model id and the truth id for the training sample - array[number_of_samples]
# (4) loss_function_value_for_training_sample - the quadratic loss function for the training sample
#
#---------------------------------------------------------------------------------------------------------------------------------------------------

def compute_f_for_training_sample( alpha_coefficients , decision_stumps_for_training_samples , sample_classid ) :
    f_terms_for_training_sample             = np.dot( decision_stumps_for_training_samples , alpha_coefficients )
    f_value_for_training_sample             = np.sum( f_terms_for_training_sample )
    model_minus_truth_for_training_sample   = f_terms_for_training_sample - sample_classid
    loss_function_value_for_training_sample = np.sum( model_minus_truth_for_training_sample ** 2 )
    return ( f_terms_for_training_sample , f_value_for_training_sample , model_minus_truth_for_training_sample , loss_function_value_for_training_sample )

#---------------------------------------------------------------------------------------------------------------------------------------------------
#
# compute_f_for_training_sample ( ) - computes the loss function for the training sample.
#
# NOTE: This function is used indirectly by the optimizer as it is invoked by the compute_fit_function( ) routine that is based to the optimizer.
# ----
#
# The input arguments are:
#
# (1) alpha_coefficients                  - the weights on the decision stump feature functions - array[number_of_features]
# (2) decision_stumps_for_training_sample - pre-computed results of the decision stumps applied to each training sample - array[samples,number_of_features]
# (3) sample_classid                      - the 'true' class id for each training sample - array[number_of_samples]
# 
# and the returned output is:
#
# (1) loss_function_value_for_training_sample - the quadratic loss function for the training sample
#
#---------------------------------------------------------------------------------------------------------------------------------------------------

def compute_loss_function_for_training_sample( alpha_coefficients , decision_stumps_for_training_samples , sample_classid ) :
    f_terms_for_training_sample             = np.dot( decision_stumps_for_training_samples , alpha_coefficients )
    f_value_for_training_sample             = np.sum( f_terms_for_training_sample )
    model_minus_truth_for_training_sample   = f_terms_for_training_sample - sample_classid
    loss_function_value_for_training_sample = np.sum( model_minus_truth_for_training_sample ** 2 )
    return loss_function_value_for_training_sample

#---------------------------------------------------------------------------------------------------------------------------------------------------
# compute_l1_regularizer( ) : regularizer function computed as the sum of the absolute value of the alpha_coefficients.
#---------------------------------------------------------------------------------------------------------------------------------------------------

def compute_l1_regularizer( alpha_coefficients ) :
    return np.sum( np.fabs( alpha_coefficients ) )

#---------------------------------------------------------------------------------------------------------------------------------------------------
# compute_l2_regularizer( ) : regularizer function computed as the sum of the quadratic value of the alpha_coefficients.
#---------------------------------------------------------------------------------------------------------------------------------------------------

def compute_l2_regularizer( alpha_coefficients ) :
    return np.sum( alpha_coefficients*alpha_coefficients )

#---------------------------------------------------------------------------------------------------------------------------------------------------
#
# compute_fit_function( ) - for the optimizer, add the loss function and a regularizer.
#
# NOTE: As this function is used by the optimizer, the first parameters must be the alpha_coefficients (what is being varied when searching for a
# ----  minimum); the other parameters can be used in whatever order is desired as they are spectators.
#
# The input arguments are:
#
# (1) alpha_coefficients                  - the weights on the decision stump feature functions - array[number_of_features]
# (2) decision_stumps_for_training_sample - pre-computed results of the decision stumps applied to each training sample - array[samples,number_of_features]
# (3) sample_classid                      - the 'true' class id for each training sample - array[number_of_samples]
# 
# and the returned output is:
#
# (1) loss_function_value_for_training_sample - the quadratic loss function for the training sample
#
#---------------------------------------------------------------------------------------------------------------------------------------------------

def compute_fit_function( alpha_coefficients , decision_stumps_for_training_samples , sample_classid ) :
    loss        = compute_loss_function_for_training_sample( alpha_coefficients , decision_stumps_for_training_samples , sample_classid )
    regularizer = compute_l2_regularizer( alpha_coefficients )
    return loss + regularizer

#---------------------------------------------------------------------------------------------------------------------------------------------------
#
# compute_f_for_single_sample( ) : computes the value of the model function, f, for the input sample.
#
# The input arguments are:
#
# (1) number_of_features        - the number of decision stump features (could be obtained from alpha_coefficients.size)
# (2) alpha_coefficients        - the weights on the decision stump feature functions - array[number_of_features]
# (3) single_sample             - sample[0,0] - x component of the sample / sample[0,1] - the y component of the sample
#                                 NOTE: array structure should be np.array( [ [ 0.0 , 0.0 ] ] ) so as routine could process more than one single sample
# (4) omega_components          - the axis on which each decision stump acts (here, either =0 for x-axis for =1 for y-axis) - array[number_of_features]
# (5) omega_thresholds_to_use   - the thresholds to apply in each decision stump - array[number_of_features]
#
#---------------------------------------------------------------------------------------------------------------------------------------------------

def compute_f_for_single_sample( single_sample , number_of_features , omega_components , omega_thresholds_to_use , alpha_coefficients ) :

#   print single_sample
    decision_stumps_for_single_sample = np.zeros( shape=(number_of_features) , dtype=float )
    for KK in feature_indices :
        decision_stumps_for_single_sample[KK] = np.sign( single_sample[0,omega_components[KK]] - omega_thresholds_to_use[KK] )
    f_terms_for_single_sample = decision_stumps_for_single_sample * alpha_coefficients
    f_value_for_single_sample = np.sum( f_terms_for_single_sample )
    return ( f_terms_for_single_sample , f_value_for_single_sample , decision_stumps_for_single_sample )

#---------------------------------------------------------------------------------------------------------------------------------------------------
#
# compute_purity_distribution( ) : compute the efficiency of a cut applied to the model function f for the samples of a specified classid; that
#                                  sample is either the training sample or the validation sample ... typically the latter.  If the specified classid
#                                  is +1, then it is the signal efficiency; if the specified classid is -1, it is the background efficiency.
#
# The input arguments are:
#
# (1) this_sample_f_values      - the values for the decision stump model function when applied to each element in the sample - array[number_of_samples]
# (2) this_sample_classids      - the true classid for the sample - array[number_of_samples]
# (3) specified_classid         - the specified classid: =1 for signal ... =-1 for background
#
# The returned otput is:
#
# (1) pdf_bin_centers                            - the center points for the cut bin
# (2) this_sample_f_values_for_signal_purity_cdf - the efficiencies at that cut bin
#
#---------------------------------------------------------------------------------------------------------------------------------------------------

def compute_purity_distribution( this_sample_f_values , this_sample_classids , specified_classid ) :

    number_of_bins  = 100
    min_f_value     = min( this_sample_f_values )
    max_f_value     = max( this_sample_f_values )
    pdf_binwidth    = ( max_f_value - min_f_value ) / number_of_bins;
    pdf_bins        = np.arange( min_f_value , max_f_value+(pdf_binwidth/2.0) , pdf_binwidth )
#   print "len( pdf_bins )        = " , len( pdf_bins )
#   print "--> min_f_value = %f - pdf_bins[first = %d ] = %f" % ( min_f_value , 0 , pdf_bins[0] )
#   print "--> max_f_value = %f - pdf_bins[last  = %d ] = %f" % ( max_f_value , len(pdf_bins)-1 , pdf_bins[len(pdf_bins)-1] )
#   pdf_bin_centers = pdf_bins + ( pdf_binwidth / 2.0 )
    this_sample_f_values_bin_indices = np.digitize( np.trim_zeros( this_sample_f_values*(this_sample_classids==specified_classid) ) , pdf_bins )
#   print "len( this_sample_f_values_bin_indices ) = " , len( this_sample_f_values_bin_indices )
    this_sample_f_values_for_signal_classid_pdf = np.bincount( this_sample_f_values_bin_indices , minlength=(len(pdf_bins)+1) )
#   print "len( this_sample_f_values_for_signal_classid_pdf ) = " , len( this_sample_f_values_for_signal_classid_pdf )
    this_sample_f_values_for_signal_classid_cdf = np.cumsum( this_sample_f_values_for_signal_classid_pdf[:-1] , dtype='float' ) / \
                                                  np.sum( this_sample_f_values_for_signal_classid_pdf[:-1] ) # could use last bin value here, instead of np.sum()
#   print "len( this_sample_f_values_for_signal_classid_cdf ) = " , len( this_sample_f_values_for_signal_classid_cdf )
    this_sample_f_values_for_signal_purity_cdf = 1.0 - this_sample_f_values_for_signal_classid_cdf

    return( pdf_bins        , this_sample_f_values_for_signal_purity_cdf )
#   return( pdf_bin_centers , this_sample_f_values_for_signal_purity_cdf )

#---------------------------------------------------------------------------------------------------------------------------------------------------
#
# compute_contamination_distribution( ) : compute the contamination for a cut applied to the model function f for the samples of a specified classid;
#                                         that sample is either the training sample or the validation sample ... typically the latter.  If the
#                                         specified classid is +1, then it is the signal efficiency; if the specified classid is -1, it is the
#                                         background efficiency.
#
# The input arguments are:
#
# (1) this_sample_f_values      - the values for the decision stump model function when applied to each element in the sample - array[number_of_samples]
# (2) this_sample_classids      - the true classid for the sample - array[number_of_samples]
# (3) specified_classid         - the specified classid: =1 for signal ... =-1 for background
#
# The returned otput is:
#
# (1) pdf_bin_centers                                      - the center points for the cut bin
# (2) this_sample_f_values_for_background_contaminatio_cdf - the contamination at that cut bin (as a fraction ... x100 to get to percentages)
#
#---------------------------------------------------------------------------------------------------------------------------------------------------

def compute_contamination_distribution( this_sample_f_values , this_sample_classids , specified_classid ) :

    number_of_bins  = 100
    min_f_value     = min( this_sample_f_values )
    max_f_value     = max( this_sample_f_values )
    pdf_binwidth    = ( max_f_value - min_f_value ) / number_of_bins;
    pdf_bins        = np.arange( min_f_value , max_f_value+(pdf_binwidth/2.0) , pdf_binwidth )
#   print "len( pdf_bins )        = " , len( pdf_bins )
#   print "--> min_f_value = %f - pdf_bins[first = %d ] = %f" % ( min_f_value , 0 , pdf_bins[0] )
#   print "--> max_f_value = %f - pdf_bins[last  = %d ] = %f" % ( max_f_value , len(pdf_bins)-1 , pdf_bins[len(pdf_bins)-1] )
#   pdf_bin_centers = pdf_bins + ( pdf_binwidth / 2.0 )
#   print "len( pdf_bin_centers ) = " , len( pdf_bin_centers )
    this_sample_f_values_bin_indices = np.digitize( np.trim_zeros( this_sample_f_values*(this_sample_classids!=specified_classid) ) , pdf_bins )
#   print "len( this_sample_f_values_bin_indices ) = " , len( this_sample_f_values_bin_indices )
    this_sample_f_values_for_background_classid_pdf = np.bincount( this_sample_f_values_bin_indices , minlength=(len(pdf_bins)+1) )
#   print "len( this_sample_f_values_for_background_classid_pdf ) = " , len( this_sample_f_values_for_background_classid_pdf )
    this_sample_f_values_for_background_classid_cdf = np.cumsum( this_sample_f_values_for_background_classid_pdf[:-1] , dtype='float' ) / \
                                                      np.sum( this_sample_f_values_for_background_classid_pdf[:-1] ) # could use last bin value here, instead of np.sum()
#   print "len( this_sample_f_values_for_background_classid_cdf ) = " , len( this_sample_f_values_for_background_classid_cdf )
#   if ( len( this_sample_f_values_for_background_classid_cdf ) > number_of_bins ) :
#       this_sample_f_values_for_background_contamination_cdf = 1.0 - this_sample_f_values_for_background_classid_cdf[:-1]
#   else :
    this_sample_f_values_for_background_contamination_cdf = 1.0 - this_sample_f_values_for_background_classid_cdf

    return( pdf_bins        , this_sample_f_values_for_background_contamination_cdf )
#   return( pdf_bin_centers , this_sample_f_values_for_background_contamination_cdf )

#---------------------------------------------------------------------------------------------------------------------------------------------------
# make_signal_purity_and_background_contamination_vs_f_value_cut_plot( ) : plot the signal efficiency and background contamination -vs- the model
#                                                                          function f value where 'blue' and 'red' are the signal efficiency and
#                                                                          background contamination curves, respectively.
#---------------------------------------------------------------------------------------------------------------------------------------------------

def make_signal_purity_and_background_contamination_vs_f_value_cut_plot( title_label , contamination_bin_centers , contamination_values , purity_bin_centers , purity_values ) :
 
    debug_make_signal_purity_and_background_contamination_vs_f_value_cut_plot = False # =True, produce some debug output ... =False, keep quiet

    this_figure = plt.figure( )
    this_plot_axis_object = this_figure.add_subplot( 111 )
    this_plot_axis_object.grid( )
    this_plot_axis_object.set_title( title_label )

    if ( debug_make_signal_purity_and_background_contamination_vs_f_value_cut_plot ) :
       print "--> contamination info: " , len( contamination_bin_centers ) , " - " , len( contamination_values*100.0 )
       print "-->        purity info: " , len( purity_bin_centers        ) , " - " , len( purity_values*100.0        )
 
    this_plot_axis_object.plot( contamination_bin_centers , contamination_values*100.0 , color='r' , label='background' )
    this_plot_axis_object.plot( purity_bin_centers        , purity_values*100.0        , color='b' , label='signal'     )
    this_plot_axis_object.set_xlabel( 'cut value for model function f (unitless)' )
    this_plot_axis_object.set_ylabel( 'signal efficiency or background contamination (percentage)' )
#   this_legend = this_plot_axis_object.legend( loc='upper right' , shadow=False )
    this_figure.show( )

    return ( this_figure , this_plot_axis_object )

#---------------------------------------------------------------------------------------------------------------------------------------------------
# make_roc_curve_plot : roc plot for the input contaimination and efficiency vectors.
#---------------------------------------------------------------------------------------------------------------------------------------------------

def make_roc_curve_plot( contamination_bin_centers , contamination_values , purity_bin_centers , purity_values ) :
 
    debug_make_roc_curve_plot_routine = False # =True, generate some debug info as plot is made ... =False, be quiet

    this_figure = plt.figure( )
    this_plot_axis_object = this_figure.add_subplot( 111 )
    this_plot_axis_object.set_title( 'ROC' )
    this_plot_axis_object.grid( )

    if ( debug_make_roc_curve_plot_routine ) :
       print "--> contamination info: " , len( contamination_bin_centers ) , " - " , len( contamination_values*100.0 )
       print "-->    -> bin centers - " , contamination_bin_centers
       print "-->    ->      values - " , contamination_values*100
       print "--> ...... purity info: " , len( purity_bin_centers        ) , " - " , len( purity_values*100.0        )
       print "-->    -> bin centers - " , purity_bin_centers
       print "-->    ->      values - " , purity_values*100
 
    this_plot_axis_object.plot( contamination_values*100.0 , purity_values*100.0 , color='r' , label='Random Kitchen' )
    this_plot_axis_object.set_xlabel( 'background contamination (percentage)' )
    this_plot_axis_object.set_ylabel( 'signal efficiency (percentage)' )
    this_legend = this_plot_axis_object.legend( loc='lower right' , shadow=False )
    this_figure.show( )

    return ( this_figure , this_plot_axis_object )

#---------------------------------------------------------------------------------------------------------------------------------------------------
# make_roc_comparison_curve_plot : roc plot for the kitchen sink and BDT from the respective contamination and efficiency vectors.
#---------------------------------------------------------------------------------------------------------------------------------------------------

def make_roc_comparison_curve_plot( contamination_bin_centers_1 , contamination_values_1 , purity_bin_centers_1 , purity_values_1 , \
                                    contamination_bin_centers_2 , contamination_values_2 , purity_bin_centers_2 , purity_values_2 ) :
 
    debug_make_roc_curve_comparision_plot_routine = False # =True, generate debug info as plot is made ... =False, be quiet

    this_figure = plt.figure( )
    this_plot_axis_object = this_figure.add_subplot( 111 )
    this_plot_axis_object.set_title( 'ROC' )
    this_plot_axis_object.grid( )
    major_ticks = np.arange( 0 , 101 , 10 )
    minor_ticks = np.arange( 0 , 101 ,  5 )
    this_plot_axis_object.set_xticks( major_ticks )
    this_plot_axis_object.set_xticks( minor_ticks , minor=True )
    this_plot_axis_object.set_yticks( major_ticks )
    this_plot_axis_object.set_yticks( minor_ticks , minor=True ) 
    
    if ( debug_make_roc_curve_comparision_plot_routine ) :
       print "--> contamination info: " , len( contamination_bin_centers ) , " - " , len( contamination_values*100.0 )
       print "-->    -> bin centers - " , contamination_bin_centers
       print "-->    ->      values - " , contamination_values*100
       print "--> ...... purity info: " , len( purity_bin_centers        ) , " - " , len( purity_values*100.0        )
       print "-->    -> bin centers - " , purity_bin_centers
       print "-->    ->      values - " , purity_values*100
 
    this_plot_axis_object.plot( contamination_values_1*100.0 , purity_values_1*100.0 , color='r' , label='Random Kitchen' )
    this_plot_axis_object.plot( contamination_values_2*100.0 , purity_values_2*100.0 , color='b' , label='BDT'            )
    this_plot_axis_object.set_xlabel( 'background contamination (percentage)' )
    this_plot_axis_object.set_ylabel( 'signal efficiency (percentage)' )
    this_legend = this_plot_axis_object.legend( loc='lower right' , shadow=False )
    this_figure.show( )

    return ( this_figure , this_plot_axis_object )

#---------------------------------------------------------------------------------------------------------------------------------------------------
# make_signal_vs_background_overlay_scatter_plot( ) : make a scatter plot of the dataset with signal as 'blue' points and background as 'red' ones.
#---------------------------------------------------------------------------------------------------------------------------------------------------

def make_signal_vs_background_overlay_scatter_plot( this_sample , this_sample_classid ) :

    this_figure            = plt.figure( )
    this_plot_axis_object  = this_figure.add_subplot( 111 )
    this_plot_axis_object.grid( )

    x = np.trim_zeros( this_sample[:,0]*(this_sample_classid==1) )   # signal points in blue
    y = np.trim_zeros( this_sample[:,1]*(this_sample_classid==1) )
    this_plot_axis_object.scatter( x , y , c=('b') , alpha=0.50 )

    x = np.trim_zeros( this_sample[:,0]*(this_sample_classid==-1) )  # background points in red
    y = np.trim_zeros( this_sample[:,1]*(this_sample_classid==-1) )
    this_plot_axis_object.scatter( x , y , c=('r') , alpha=0.25 )

    this_plot_axis_object.set_xlim( [ -10.0 , 10.0 ] )
    this_plot_axis_object.set_ylim( [ -10.0 , 10.0 ] )

    this_figure.show( )

    return ( this_figure , this_plot_axis_object )

#---------------------------------------------------------------------------------------------------------------------------------------------------
# make_signal_vs_background_overlay_histogram_plot( ) : make a plot with an overlay of the model function values for the signal (blue) and
#                                                       background (red) samples.
#---------------------------------------------------------------------------------------------------------------------------------------------------

def make_signal_vs_background_overlay_histogram_plot( validation_sample_f_value , validation_sample_classid ) :

    this_figure = plt.figure( )
    this_plot_axis_object = this_figure.add_subplot( 111 )
    this_plot_axis_object.grid( )
    this_plot_x_axis_definition = np.arange( -4.0 , 4.0 , (1.0/20.0) )
    ( n_signal     , bins_signal     , patches_signal )     = this_plot_axis_object.hist( np.trim_zeros( validation_sample_f_value*(validation_sample_classid==1) )  , # signal
                                                                                          bins=this_plot_x_axis_definition ,
                                                                                          facecolor='blue' , alpha=0.20 )
#   print "--> n_signal    = " , n_signal
#   print "--> bins_signal = " , bins_signal
    ( n_background , bins_background , patches_background ) = this_plot_axis_object.hist( np.trim_zeros( validation_sample_f_value*(validation_sample_classid==-1) ) , # background
                                                                                          bins=this_plot_x_axis_definition ,
                                                                                          facecolor='red' , alpha=0.40 )
    this_figure.show( )

    return ( this_figure , this_plot_axis_object )

#---------------------------------------------------------------------------------------------------------------------------------------------------
# dump_f_info_for_training_sample_order_sample_by_feature( ) : DEBUG ROUTINE - dump the current information about the training sample and the model
#                                                              function f.
#---------------------------------------------------------------------------------------------------------------------------------------------------

def dump_f_info_for_training_sample_order_sample_by_feature( alpha_coefficients ,
                                                             sample , sample_classid ,
                                                             omega_components , omega_thresholds_to_use ,
                                                             decision_stumps_for_training_samples ) :
    for KK in feature_indices :
        for II in sample_indices :
            if ( omega_components[KK] == 0 ) :
               print "sample %2d - feature %2d - sample[%2d] = ( * %9.6f * ,   %9.6f   ) - threshold %9.6f - class: stump %2d / true %2d - alpha %9.6f" % ( II , KK , II ,
                             sample[II,0] , sample[II,1] ,
                             omega_thresholds_to_use[KK] ,
                             decision_stumps_for_training_samples[II,KK] , sample_classid[II] ,
                             alpha_coefficients[KK]
                     )
            else :
               print "sample %2d - feature %2d - sample[%2d] = (   %9.6f   , * %9.6f * ) - threshold %9.6f - class: stump %2d / true %2d - alpha %9.6f" % ( II , KK , II ,
                             sample[II,0] , sample[II,1] ,
                             omega_thresholds_to_use[KK] ,
                             decision_stumps_for_training_samples[II,KK] , sample_classid[II] ,
                             alpha_coefficients[KK]
                     )
        print "-----------------------------------------------------------------------------------------------------------------------------------------------------------------"

    return

#---------------------------------------------------------------------------------------------------------------------------------------------------
# dump_f_info_for_training_sample_order_feature_by_sample( ) : DEBUG ROUTINE - dump the current information about the training sample and the model
#                                                              function f.
#---------------------------------------------------------------------------------------------------------------------------------------------------

def dump_f_info_for_training_sample_order_feature_by_sample( alpha_coefficients , sample , sample_classid ,
                                                             omega_components , omega_thresholds_to_use ,
                                                             decision_stumps_for_training_samples , f_terms_for_training_sample ) :
    for II in sample_indices :
        for KK in feature_indices :
            if ( omega_components[KK] == 0 ) :
               print "sample %2d - feature %2d - sample[%2d] = ( * %9.6f * ,   %9.6f   ) - threshold %9.6f - class: stump %2d / true %2d - alpha %9.6f - f_term %9.6f" % ( II , KK , II ,
                             sample[II,0] , sample[II,1] ,
                             omega_thresholds_to_use[KK] ,
                             decision_stumps_for_training_samples[II,KK] , sample_classid[II] ,
                             alpha_coefficients[KK] ,
                             alpha_coefficients[KK]*decision_stumps_for_training_samples[II,KK]
                     )
            else :
               print "sample %2d - feature %2d - sample[%2d] = (   %9.6f   , * %9.6f * ) - threshold %9.6f - class: stump %2d / true %2d - alpha %9.6f - f_term %9.6f" % ( II , KK , II ,
                             sample[II,0] , sample[II,1] ,
                             omega_thresholds_to_use[KK] ,
                             decision_stumps_for_training_samples[II,KK] , sample_classid[II] ,
                             alpha_coefficients[KK] ,
                             alpha_coefficients[KK]*decision_stumps_for_training_samples[II,KK]
                     )
        print "sample %2d -             " % ( II ) ,
        print "                                                                                                                        sum %9.6f" % ( f_terms_for_training_sample[II] )
        print "-----------------------------------------------------------------------------------------------------------------------------------------------------------------"

    return

#---------------------------------------------------------------------------------------------------------------------------------------------------
# dump_decision_stump_info_for_single_sample( ) : DEBUG ROUTINE - dump the current information about a single sample and the model function f.
#---------------------------------------------------------------------------------------------------------------------------------------------------

def dump_decision_stump_info_for_single_sample( single_sample , decision_stumps_for_single_sample ,
                                                feature_indices , alpha_coefficients , omega_components , omega_thresholds_to_use ) :

    for KK in feature_indices :
        if ( omega_components[KK] == 0 ) :
           print "single_sample - feature %2d - single_sample[0,0] = ( * %9.6f * ,   %9.6f   ) - threshold %9.6f - data_stump_value %2d - alpha %9.6f / f_term %9.6f" % ( KK ,
                 single_sample[0,0] , single_sample[0,1] ,
                 omega_thresholds_to_use[KK] ,
                 decision_stumps_for_single_sample[KK] ,
                 alpha_coefficients[KK] ,
                 f_terms_for_single_sample[KK]
           )
        else :
           print "single_sample - feature %2d - single_sample[0,0] = (   %9.6f   , * %9.6f * ) - threshold %9.6f - data_stump_value %2d - alpha %9.6f / f_term %9.6f" % ( KK ,
                 single_sample[0,0] , single_sample[0,1] ,
                 omega_thresholds_to_use[KK] ,
                 decision_stumps_for_single_sample[KK] ,
                 alpha_coefficients[KK] ,
                 f_terms_for_single_sample[KK]
           )

    print "------------- - f_term array = " , f_terms_for_single_sample
    print "------------- -          sum = " , f_value_for_single_sample

    return

# ====================================================================================================================================================
# =                                                                                                                                                  =
# =                                                    S T A R T   O F   M A I N   P R O G R A M                                                     =
# =                                                                                                                                                  =
# ====================================================================================================================================================

debug_program_steps     = True  # =True, generate some output as program progresses          ... =False, keep it quiet
debug_main              = False # =True, generate debug output from main code                ... =False, keep quiet
debug_training          = False # =True, generate debug output as training sample processed  ... =False, keep quiet
debug_optimization_step = True  # =True, generate debug output when the optimization is done ... =False, keep quiet
debug_validation        = False # =True, dump the validation data to stdout                  ... =False, keep quiet

debug_bdt               = False # =True, generate debug output from BDT section of code      ... =False, keep quiet

#
# -- generate a training dataset : background (classid = -1) - 2d normal position at (  1.0 ,  1.0 ) with variance of 4.0, i.e. std_dev=2.0
#                                  signal     (classid = +1) - 2d normal position at ( -1.0 , -1.0 ) with variance of 4.0, i.e. std_dev=2.0

if ( debug_program_steps ) :
   print "--> generating training dataset"

number_of_background_points = 100
number_of_signal_points     = 100

( number_of_training_points , sample_indices , sample , sample_x , sample_y , sample_classid , \
               sample_x_min , sample_x_max , sample_y_min , sample_y_max , sample_max , sample_min ) = generate_dataset( number_of_background_points , number_of_signal_points )

if ( debug_main or debug_training ) :
   for II in sample_indices :
       print "sample : sample[%2d] = ( %9.6f , %9.6f ) - sample_classid = %d" % ( II , sample[II,0] , sample[II,1] , sample_classid[II] )
   print "max info : x_max = %9.6f , y_max = %9.6f -> max = %9.6f" % ( sample_x_max , sample_y_max , sample_max )
   print "min info : x_min = %9.6f , y_min = %9.6f -> min = %9.6f" % ( sample_x_min , sample_y_min , sample_min )

# -- generate the feature : data_stumps ... randomly choose x or y component for stump and assign threshold values randomly
#
#    NOTE: instead of scaling the input data to +/- 1, scale the thresholds.  The behavior is equivalent, just that looking at the
#    ----  data and data stumps is easier without the normalization.

if ( debug_program_steps ) :
   print "--> generating features"

number_of_features      = 50
feature_indices         = np.arange( 0 , number_of_features , 1 )
omega_components        = ( np.random.uniform( 0.0 , 1.0 , size=number_of_features ) > 0.5 ) * 1.0  # either 0 or 1, choosen randomly
omega_thresholds        = ( np.random.uniform( 0.0 , 1.0 , size=number_of_features ) )              # thresholds which would be applied to normalized data
omega_thresholds_to_use = omega_thresholds * ( sample_max - sample_min ) + sample_min               # but, instead of normalizing data, scale & shift the thresholds

# -- generate the decision stump results for each training sample and each feature function
#
#    The stump array is a 2d array with dimensions (number_of_samples x number_of_features).
#    The value of the function for sample[ii] is then np.sum( alpha_coefficient*stump[II,:] )
#    as along alpha_coefficient is a 1d array with dimension (number_of_features)
#

if ( debug_program_steps ) :
   print "--> precomputing decision stump results for the training sample"

decision_stumps_for_training_samples = generate_decision_stumps_for_training_sample( number_of_training_points , sample ,
                                                                                     number_of_features , omega_components , omega_thresholds_to_use )

if ( debug_training ) :
   ( training_scatter_plot_figure , training_scatter_plot_axis_object ) = make_signal_vs_background_overlay_scatter_plot( sample , sample_classid )

# -- generate an initial guess for the alpha_coefficient from an uniform random distribution

alpha_coefficients = np.random.uniform( 0.0 , 1.0 , size=number_of_features )

# -- compute f terms, f value, and loss_function for the training sample

if ( debug_program_steps ) :
   print "--> computing f_terms and f_values for the training sample using random guess for alpha_coefficients (this is really for debugging purposes)"
   
( f_terms_for_training_sample , f_value_for_training_sample , model_minus_truth_for_training_sample , loss_function_value_for_training_sample ) = \
                                                          compute_f_for_training_sample( alpha_coefficients , decision_stumps_for_training_samples , sample_classid )

if ( debug_main or debug_training ) :
   print "f_terms           = " , f_terms_for_training_sample
   print "f_value           = " , f_value_for_training_sample
   print "model_minus_truth = " , model_minus_truth_for_training_sample
   print "loss              = " , loss_function_value_for_training_sample
   dump_f_info_for_training_sample_order_feature_by_sample( alpha_coefficients , sample , sample_classid , \
                                                            omega_components , omega_thresholds_to_use , \
                                                            decision_stumps_for_training_samples , f_terms_for_training_sample )

# -- do the optimization and find 'good' values for the alpha_coefficients

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# def compute_f_for_training_sample( alpha_coefficients , decision_stumps_for_training_samples , sample_classid ) :
#     f_terms_for_training_sample             = np.dot( decision_stumps_for_training_samples , alpha_coefficients )
#     f_value_for_training_sample             = np.sum( f_terms_for_training_sample )
#     model_minus_truth_for_training_sample   = f_terms_for_training_sample - sample_classid
#     loss_function_value_for_training_sample = np.sum( model_minus_truth_for_training_sample ** 2 )
#     return ( f_terms_for_training_sample , f_value_for_training_sample , loss_function_value_for_training_sample )
#
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# def compute_loss_function_for_training_sample( alpha_coefficients , decision_stumps_for_training_samples , sample_classid ) :
#     f_terms_for_training_sample             = np.dot( decision_stumps_for_training_samples , alpha_coefficients )
#     f_value_for_training_sample             = np.sum( f_terms_for_training_sample )
#     model_minus_truth_for_training_sample   = f_terms_for_training_sample - sample_classid
#     loss_function_value_for_training_sample = np.sum( model_minus_truth_for_training_sample ** 2 )
#     return loss_function_value_for_training_sample
#
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# def compute_fit_function( alpha_coefficients , decision_stumps_for_training_samples , sample_classid ) :
#     loss        = compute_loss_function_for_training_sample( alpha_coefficients , decision_stumps_for_training_samples , sample_classid )
#     regularizer = compute_l2_regularizer( alpha_coefficients )
#     return loss + regularizer
#
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

if ( debug_program_steps ) :
   print "--> doing the optimization for alpha_coefficients using decision_stumps computed from training sample"

start_alpha_coefficients = np.random.uniform( 0.0 , 1.0 , size=number_of_features )
if ( debug_main or debug_optimization_step ) :
   print "---> start_alpha_coefficients = "
   print start_alpha_coefficients

fit_results = optimizer.minimize( compute_fit_function ,
                                  start_alpha_coefficients , args=( decision_stumps_for_training_samples , sample_classid ) ,
                                  method='powell' , options={ 'maxiter':2000000 , 'maxfev':2000000 , 'ftol': 1.0e-8 , 'disp': True } ) # tolerance = 1.0e-8

if ( debug_main or debug_optimization_step ) :
   print fit_results

alpha_coefficients_from_fit = fit_results['x'] # recover the final values for the alpha_coefficients

# -- compute the model function for a single_sample (useful when debugging the code)

print "---------------------------------------------------------------------------------------------------------------------------------------------------------------"

if ( debug_program_steps ) :
   print "--> evaluate fitted model function for single sample at (0.0, 0.0) [again, really a debugging operation]"

single_sample = np.array( [ [ 0.0 , 0.0 ] ] )
if ( len( alpha_coefficients_from_fit ) > 0 ) :
   if ( debug_main or debug_validation ) :
      print "----> USING FIT RESULT ALPHA_COEFFICIENTS"
   ( f_terms_for_single_sample , f_value_for_single_sample , decision_stumps_for_single_sample ) = compute_f_for_single_sample( single_sample ,
                                                                                                                                number_of_features ,
                                                                                                                                omega_components ,
                                                                                                                                omega_thresholds_to_use ,
                                                                                                                                alpha_coefficients_from_fit )
else :
   if ( debug_main or debug_validation ) :
      print "----> USING GUESS VALUES FOR ALPHA_COEFFICIENTS"
   ( f_terms_for_single_sample , f_value_for_single_sample , decision_stumps_for_single_sample ) = compute_f_for_single_sample( single_sample ,
                                                                                                                                number_of_features ,
                                                                                                                                omega_components ,
                                                                                                                                omega_thresholds_to_use ,
                                                                                                                                alpha_coefficients )
if ( debug_main or debug_validation ) :
   dump_decision_stump_info_for_single_sample( single_sample , decision_stumps_for_single_sample ,
                                               feature_indices , alpha_coefficients , omega_components , omega_thresholds_to_use )

   
# -- painful though it is, write out f-values for a grid in (x,y) so that I can look at a nice contour plot using
#            gnuplot; sadly, the color scheme in matplotlib was really ugly and I decided to not focus on figuring
#            out how to change it to a pretty one.  Something for later.
#
#             NOTE: all files are ' ' separated ... so don't forget the dataset separator ' ' line in gnuplot!
#             ----

if ( debug_program_steps ) :
   print "--> writing model function values for a meshgrid to a file so that I make a prettier contour plot with gnuplot [a debug feature]"
   
filename = "play_with_it"

# --- first:  the sample data in one file (in case I want to overlay a scatter plot of the data on the contour plot)

outFile = open(filename+".training_sample.dat" , "w" )

for II in sample_indices :
    outFile.write( str(II) + " " + str(sample[II,0]) + " " + str(sample[II,1]) + " " + str(sample_classid[II]) )
    outFile.write( "\n" )

outFile.close( )

# --- second: the model result in another file

delta = 0.05
plot_x_range = np.arange( sample_x_min-1.0 , sample_x_max+1.0 , delta )
plot_y_range = np.arange( sample_y_min-1.0 , sample_y_max+1.0 , delta )

# ---         A - little summary file to know the meshgrid range and sizes

outFile = open( filename+".model.meshgrid.info" , "w" )
outFile.write( "x_min " + str(sample_x_min-1.0) + " " + "x_max " + str(sample_x_max+1.0) + " " + "delta " + str( delta ) + " " + "npoints " + str( len(plot_x_range) ) )
outFile.write( "\n" )
outFile.write( "y_min " + str(sample_y_min-1.0) + " " + "y_max " + str(sample_y_max+1.0) + " " + "delta " + str( delta ) + " " + "npoints " + str( len(plot_y_range) ) )
outFile.write( "\n" )
outFile.close( )

# ---         B - the model result in another file

outFile = open( filename+".model.meshgrid.dat"  , "w" )

for POINT_x in plot_x_range :
    for POINT_y in plot_y_range :
        single_sample = np.array( [ [ POINT_x , POINT_y ] ] )
        ( f_terms_for_single_sample , f_value_for_single_sample , decision_stumps_for_single_sample ) = compute_f_for_single_sample( single_sample ,
                                                                                                                                     number_of_features ,
                                                                                                                                     omega_components ,
                                                                                                                                     omega_thresholds_to_use ,
                                                                                                                                     alpha_coefficients_from_fit )
        outFile.write( str(POINT_x) + " " + str(POINT_y) + " " + str(f_value_for_single_sample) )
        outFile.write( "\n" )

outFile.close( )

# -- generate the validation dataset

if ( debug_program_steps ) :
   print "--> generating validation dataset"

number_of_validation_background_points = 20000
number_of_validation_signal_points     = 20000

( number_of_validation_points , validation_sample_indices , \
  validation_sample , validation_sample_x , validation_sample_y , validation_sample_classid , \
  validation_sample_x_min , validation_sample_x_max , \
  validation_sample_y_min , validation_sample_y_max , \
  validation_sample_max   , validation_sample_min   ) = generate_dataset( number_of_validation_background_points , number_of_validation_signal_points )

# -- make a scatter plot of the validation dataset

if ( debug_program_steps ) :
   print "--> making scatter plot for validation dataset"

( validation_scatter_plot_figure , validation_scatter_plot_axis_object ) = make_signal_vs_background_overlay_scatter_plot( validation_sample , validation_sample_classid )

# -- compute the f-values for all validation samples
#
#    okay, I should do this without the loop here as numpy does vectors.  But, for now, let's worry about the
#    result and comparing it to a bdt and, if need be, fix this detail later.

if ( debug_program_steps ) :
   print "--> compute fitted model function values for the validation dataset"
   
validation_sample_f_value = np.zeros( shape=(number_of_validation_points) )
for II in validation_sample_indices :
    single_sample = np.array( [ [ validation_sample[II,0] , validation_sample[II,1] ] ] )
    ( f_terms_for_single_sample , f_value_for_single_sample , decision_stumps_for_single_sample ) = compute_f_for_single_sample( single_sample ,
                                                                                                                                 number_of_features ,
                                                                                                                                 omega_components ,
                                                                                                                                 omega_thresholds_to_use ,
                                                                                                                                 alpha_coefficients_from_fit )
    validation_sample_f_value[II] = f_value_for_single_sample

# -- write the validation sample
#    (as above, in case I want to overlay these points on the contour plot of the model function made with gnuplot)

outFile = open(filename+".validation_sample.dat","w")

for II in validation_sample_indices :
    outFile.write( str(II) + " " +
                   str(validation_sample[II,0]) + " " + str(validation_sample[II,1]) + " " +
                   str(validation_sample_classid[II]) + " " + str(validation_sample_f_value[II]) )
    outFile.write( "\n" )

outFile.close( )

# -- make an overlay plot of the signal (blue) and background (red) distributions for model f(x,y)

if ( debug_program_steps ) :
   print "--> make an overlay plot of the distributions of the validation sample -vs- fitted model function value"
   
( validation_sample_f_distribution_figure , validation_sample_f_distribution_axis_object ) = \
                                            make_signal_vs_background_overlay_histogram_plot( validation_sample_f_value , validation_sample_classid )


# -- compute the signal efficiency and background contamination for cuts on the model function

if ( debug_program_steps ) :
   print "--> compute the signal efficiency and background contamination for the validation dataset"

( validation_purity_bin_centers , validation_purity )               = compute_purity_distribution( validation_sample_f_value , validation_sample_classid ,  1 )       # signal
( validation_contamination_bin_centers , validation_contamination ) = compute_contamination_distribution( validation_sample_f_value , validation_sample_classid , 1 ) # 1.0-bckgrnd

# -- make the signal efficiency and background contamination plot

if ( debug_program_steps ) :
   print "--> make a plot of the signal efficiency and background contamination for the validation dataset"

( purity_and_background_figure , purity_and_background_plot_axis_object ) = make_signal_purity_and_background_contamination_vs_f_value_cut_plot(              \
                                                                                            "Random Sinks" , \
                                                                                            validation_contamination_bin_centers , validation_contamination , \
                                                                                            validation_purity_bin_centers        , validation_purity        )

# -- make the ROC curve for the model

if ( debug_program_steps ) :
   print "--> make a ROC curve from the signal efficiency and background contamination for the validation dataset"

( roc_figure , roc_plot_axis_object ) = make_roc_curve_plot( validation_contamination_bin_centers , validation_contamination , \
                                                             validation_purity_bin_centers        , validation_purity        )

# -- now do a BDT for comparison to the kitchen sink model

#    1 - define the BDT object

if ( debug_program_steps ) :
   print "--> make a BDT from the training sample"

bdt = AdaBoostClassifier( DecisionTreeClassifier( max_depth = 1 ) ,
                          algorithm = "SAMME"  ,
                          n_estimators = 200   )

#    2 - give it the training sample and have it generate the BDT

bdt.fit( sample , sample_classid )

#    3 - obtain the model classid from the BDT for the validation sample

if ( debug_program_steps ) :
   print "--> obtain BDT classid info for validation sample"
   
validation_sample_bdt_model_classid = bdt.predict( validation_sample )
if ( debug_bdt ) :
   print "bdt : len( validation_sample_bdt_model_classid ) = " , len( validation_sample_bdt_model_classid )

#    4 - obtain the model decision function values from the BDT for the validation sample

if ( debug_program_steps ) :
   print "--> obtain BDT decision function values for validation sample"
   
validation_sample_bdt_f_values      = bdt.decision_function( validation_sample )
if ( debug_bdt ) :
   print "bdt : len( validation_sample_bdt_f_values )      = " , len( validation_sample_bdt_f_values )

# -- now make some plots for the BDT results

if ( debug_program_steps ) :
   print "--> make BDT signal efficiency and background contamination -vs- f-value comparison plot"

( validation_bdt_purity_bin_centers , validation_bdt_purity )               = \
                                                compute_purity_distribution( validation_sample_bdt_f_values , validation_sample_classid ,  1 )       # signal
( validation_bdt_contamination_bin_centers , validation_bdt_contamination ) = \
                                                compute_contamination_distribution( validation_sample_bdt_f_values , validation_sample_classid , 1 ) # 1.0-bckgrnd
( bdt_purity_and_background_figure , bdt_purity_and_background_plot_axis_object ) = make_signal_purity_and_background_contamination_vs_f_value_cut_plot(              \
                                                                                            "BDT" , \
                                                                                            validation_bdt_contamination_bin_centers , validation_bdt_contamination , \
                                                                                            validation_bdt_purity_bin_centers        , validation_bdt_purity        )
# -- and, finally, compare the ROC curve of the BDT to that of the kitchen sink

if ( debug_program_steps ) :
   print "--> make BDT ROC curve plot"

( roc_figure , roc_plot_axis_object ) = make_roc_comparison_curve_plot( validation_contamination_bin_centers     , validation_contamination     , \
                                                                        validation_purity_bin_centers            , validation_purity            , \
                                                                        validation_bdt_contamination_bin_centers , validation_bdt_contamination , \
                                                                        validation_bdt_purity_bin_centers        , validation_bdt_purity        )

# -- now, make some nice plots for the BDT ... this code is basically copied from an example; I just modified it
#         to meet my aims and stop it from waiting for all plots to be deleted before returning control to the
#         python interpreter.

if ( debug_program_steps ) :
   print "--> make BDT classid vs (x,y) plot with the training sample data overlaid"

plot_colors = "rb"
plot_step   = 0.02
class_names = "BS"

bdt_figure = plt.figure( figsize=(10, 5) )

# Plot the decision boundaries

bdt_axis_plot_object_1 = bdt_figure.add_subplot(121)
bdt_axis_plot_object_1.grid( )

x_min, x_max = sample[:, 0].min() - 1, sample[:, 0].max() + 1
y_min, y_max = sample[:, 1].min() - 1, sample[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

Z = bdt.predict( np.c_[ xx.ravel() , yy.ravel() ] )
Z = Z.reshape( xx.shape )
cs = bdt_axis_plot_object_1.contourf( xx , yy , Z , cmap=plt.cm.Paired )
bdt_axis_plot_object_1.axis( "tight" )

# Plot the training or the validation points

use_training = True  # = True -> training points ... =F -> use validation points
for i, n, c in zip( range(-1,2,2), class_names , plot_colors ) :
        if ( use_training ) :
           idx = np.where(sample_classid == i)
           bdt_axis_plot_object_1.scatter( sample[idx, 0], sample[idx, 1] ,
                                         c=c, cmap=plt.cm.Paired,
                                         label="Class %s" % n)
        else :
           idx = np.where(validation_sample_classid == i)
           bdt_axis_plot_object_1.scatter( validation_sample[idx, 0], validation_sample[idx, 1] ,
                                         c=c, cmap=plt.cm.Paired,
                                         label="Class %s" % n)
bdt_axis_plot_object_1.set_xlim( [ x_min , x_max ] ) 
bdt_axis_plot_object_1.set_ylim( [ y_min , y_max ] )
bdt_axis_plot_object_1.legend( loc='upper right' )
bdt_axis_plot_object_1.set_xlabel( 'x' )
bdt_axis_plot_object_1.set_ylabel( 'y' )
bdt_axis_plot_object_1.set_title( 'Decision Boundary' )
            
# Plot the two-class decision scores

if ( debug_program_steps ) :
   print "--> make BDT decision function distribution histogram plot for signal (blue) and background (red) using the validation dataset"
   
use_training = False # = True -> training points ... =F -> use validation points

if ( use_training ) :
   twoclass_output = bdt.decision_function( sample )
else :
   twoclass_output = bdt.decision_function( validation_sample )

plot_range = ( twoclass_output.min() , twoclass_output.max() )
bdt_axis_plot_object_2 = bdt_figure.add_subplot( 122 )
bdt_axis_plot_object_2.grid( )

for i, n, c in zip( range(-1,2,2) , class_names , plot_colors ) :
    if ( use_training ) :
       bdt_axis_plot_object_2.hist( twoclass_output[sample_classid == i] ,
                                    bins=10 ,
                                    range=plot_range ,
                                    facecolor=c ,
                                    label='Class %s' % n ,
                                    alpha=.5 )
    else :
       bdt_axis_plot_object_2.hist( twoclass_output[validation_sample_classid == i] ,
                                    bins=50 ,
                                    range=plot_range ,
                                    facecolor=c ,
                                    label='Class %s' % n ,
                                    alpha=.5 )
( x1 , x2 ) = bdt_axis_plot_object_2.get_xlim( )
( y1 , y2 ) = bdt_axis_plot_object_2.get_ylim( )
bdt_axis_plot_object_2.set_xlim( [ x1 , x2 ] )
bdt_axis_plot_object_2.set_ylim( [ y1 , y2 * 1.2 ] )
bdt_axis_plot_object_2.legend( loc='upper right' )
bdt_axis_plot_object_2.set_ylabel( 'Samples' )
bdt_axis_plot_object_2.set_xlabel( 'Score' )
bdt_axis_plot_object_2.set_title( 'Decision Scores' )

bdt_figure.show( )

# ====================================================================================================================================================
# =                                                                                                                                                  =
# =                                                      E N D   O F   M A I N   P R O G R A M                                                       =
# =                                                                                                                                                  =
# ====================================================================================================================================================

