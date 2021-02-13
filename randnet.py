#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 13:57:35 2020

@author: calebhallinan
"""

# Making the randnet package in R #

# importing numpy
import numpy as np
# importing random
import random
# importing powerlaw, must downlaod from github
import powerlaw 
# importing math
import math
# import skicit
from sklearn.cluster import KMeans
# downloaded this from github
# pip install -e git+https://github.com/bwlewis/irlbpy.git#egg=irlb
# had to make a slight change on line 138 deleting the x
from irlb import irlb
# for norm function
from numpy import linalg as LA
# for roc curve
import sklearn.metrics as metrics
# to check speed of functions
import timeit
# to copy matrices
from copy import deepcopy



def BlockModelGen(lamb, n, beta = 0, K = 3, rho = 0, simple = True, 
                  power = True, alpha = 5, degree_seed = None):
    """
    Description
    ----------
    Generates networks from degree corrected stochastic block model, with various options for node degree distribution
        
    Arguments
    ----------

    lambda : average node degree
    
    n : size of network
    
    beta : out-in ratio: the ratio of between-block edges over within-block edges
    
    K : number of communities
    
    w : not effective
    
    Pi : a vector of community proportion
    
    rho : proportion of small degrees within each community if the degrees are from 
        two point mass disbribution. rho >0 gives degree corrected block model. If 
        rho > 0 and simple=TRUE, then generate the degrees from two point mass 
        distribution, with rho porition of 0.2 values and 1-rho proportion of 
        1 for degree parameters. If rho=0, generate from SBM.
    
    simple : Indicator of wether two point mass degrees are used, if rho > 0. 
        If rho=0, this is not effective
    
    power : Whether or not use powerlaw distribution for degrees. If FALSE, 
        generate from theta from U(0.2,1); if TRUE, generate theta from powerlaw. 
        Only effective if rho >0, simple=FALSE.
    
    alpha : Shape parameter for powerlaw distribution.
    
    degree.seed : Can be a vector of a prespecified values for theta. Then the 
        function will do sampling with replacement from the vector to generate theta. 
        It can be used to control noise level between different configuration settings.

    Returns
    -------
    A dictionary with:
        (variable name)["A"] : the generated network adjacency matrix
        
        (variable name)["g"] : community membership
        
        (variable name)["P"] : probability matrix of the network
        
        (variable name)["theta"] : node degree parameter
        
    Author(s)
    ----------
    Tianxi Li, Elizaveta Levina, Ji Zhu
    
    """
    w = [1] * K
    Pi = 1/K
    
    P0 = np.diag(w)
    if (beta > 0):
        P0 = np.ones((K, K), dtype=np.int32)
        diag_P0 = [w_element / beta for w_element in w]
        np.fill_diagonal(P0, diag_P0)
    
    Pi_vec = [[Pi] for i in range(K)]
    P1 = lamb * P0
    P2 = np.matmul((np.transpose(Pi_vec)), P0)
    P3 = (n-1) * np.matmul(P2, Pi_vec) * (rho * 0.2 + (1-rho))**2
    P = P1/P3
    
    if (rho > 0) and (simple != True) and (power != True):
        P1 = lamb * P0
        P2 = np.matmul(((n-1) * np.transpose(Pi_vec)), P0)
        P3 = np.matmul(P2, Pi_vec) * (0.6)**2
        P = P1/P3
        
    if (rho > 0) and (simple != True) and (power == True):
        P1 = lamb * P0
        P2 = np.matmul(((n-1) * np.transpose(Pi_vec)), P0)
        P3 = np.matmul(P2, Pi_vec) * (1.285)**2
        P = P1/P3
        
    M = np.zeros((n, K), dtype=np.int32)
    membership = random.choices(range(0, K), k = n, weights = [Pi]*K)
       
    i = 0
    while i < n:
        M[i][membership[i]] = 1
        i +=1 
    MP = np.matmul(M, P)
    A_bar = np.matmul(MP, np.transpose(M))
    node_degree = [1] * n
        
    if rho > 0:
        randunif = np.random.uniform(size = n)
        if simple == True:
            j = 0
            while j < n:
                if randunif[j] < rho:
                    node_degree[j] = 0.2
                    j += 1
                else:
                    j += 1
        else:
            if power == False:
                node_degree = np.random.uniform(size = n) * 0.8 + 0.2
            else:
                MM = math.ceil(n/300)
                if degree_seed == None:
                    degree_seed = powerlaw.Power_Law(xmin=1, parameters=[alpha]).generate_random(n)
                node_degree = random.choices(degree_seed, k = n)
                
    DD = np.diag(node_degree)
    A_bar = np.matmul(DD, A_bar)
    A_bar = np.matmul(A_bar, DD)   
    A_bar = A_bar * lamb/np.mean(np.sum(A_bar, axis = 0))
    upper_index = np.triu_indices(n, k = 1)
    upper_p = A_bar[upper_index]
    upper_u = np.random.uniform(size = len(upper_p))
    upper_A = np.where(upper_u < upper_p, 1, 0)
    A = np.zeros((n, n), dtype=np.int32)
    A[upper_index] = upper_A
    A = A + np.transpose(A)
    np.fill_diagonal(A, 0)
    
    # return statement in dictionary form
    dic = dict()
    # generated network adjacency matrix
    dic["A"] = A
    # community membership
    dic["g"] = membership
    # probability matrix of the network
    dic["P"] = A_bar
    # node degree parameter
    dic["theta"] = node_degree
    return(dic)    



def reg_SP(A, K, tau = 1, lap = False):
    """
    Description
    ----------
    Community detection by regularized spectral clustering

    Arguments
    ----------
    A : Adjacency Matrix
    
    K : Number of Communities
    
    tau : reguarlization parameter. Default value is one. Typically set 
        between 0 and 1. If tau=0, no regularization is applied.
    
    lap : indicator. If TRUE, the Laplacian matrix for clustering. If FALSE, 
        the adjacency matrix will be used.

    Returns
    -------
    A dictionary with:
        (variable name)["cluster"] giving the cluster levels
        
        (variable name)["loss"] giving the loss of KMeans algorithm
        
    Author(s)
    ----------
    Tianxi Li, Elizaveta Levina, Ji Zhu

    """
    avg_d = np.mean(A.sum(axis = 0))
    A_tau = (A + tau*avg_d)/len(A)
    if lap != True:
        SVD = irlb(A_tau, K,  maxit=1000)
    else:
        d_tau = A_tau.sum(axis = 0)
        pre_L_tau = np.matmul(np.diag(1/np.sqrt(d_tau)), A_tau)
        L_tau = np.matmul(pre_L_tau, np.diag(1/np.sqrt(d_tau)))
        SVD = irlb(L_tau, K,  maxit=1000)
    dic = dict()
    dic["cluster"] = KMeans(n_clusters=K, max_iter = 30).fit_predict(SVD["V"][:,0:K])
    dic["loss"] = KMeans(n_clusters=K, max_iter = 30).fit(SVD["V"][:,0:K]).inertia_
    return(dic)



def reg_SSP(A, K, tau = 1, lap = False):
    """
    Description
    ----------
    Community detection by regularized spherical spectral clustering

    Arguments
    ----------
    A : Adjacency Matrix
    
    K : Number of Communities
    
    tau : reguarlization parameter. Default value is one. Typically set between 
        0 and 1. If tau=0, no regularization is applied.
    
    lap : indicator. If TRUE, the Laplacian matrix for clustering. If FALSE, 
        the adjacency matrix will be used.

    Returns
    -------
    A dictionary with:
        (variable name)["cluster"] giving the cluster levels
        
        (variable name)["loss"] giving the loss of KMeans algorithm
        
    Author(s)
    ----------
    Tianxi Li, Elizaveta Levina, Ji Zhu

    """
    
    # finding the average distance by taking mean of columns summed
    avg_d = np.mean(A.sum(axis = 0))
    A_tau = A + (tau*avg_d)/len(A)
    
    if lap != True:
        SVD = irlb(A_tau, K, maxit=1000)
        V = SVD["V"][:,0:K]
        V_norm = LA.norm(V, axis=1)
        V_normalized = np.matmul(np.diag(1/V_norm), V)
    else:
        d_tau = A_tau.sum(axis = 0)
        pre_L_tau = np.matmul(np.diag(1/np.sqrt(d_tau)), A_tau)
        L_tau = np.matmul(pre_L_tau, np.diag(1/np.sqrt(d_tau)))
        SVD = irlb(L_tau, K, maxit=1000)
        V = SVD["V"][:,0:K]
        V_norm = LA.norm(V, axis=1)
        V_normalized = np.matmul(np.diag(1/V_norm), V)
    dic = dict()
    dic["cluster"] = KMeans(n_clusters=K, max_iter = 30).fit_predict(V_normalized)
    dic["loss"] = KMeans(n_clusters=K, max_iter = 30).fit(V_normalized).inertia_
    return(dic)



def SBM_estimate(A, g):
    """
    Description
    ----------
    Estimates SBM parameters given community labels
 
    Arguments
    ----------
    A : adjacency matrix
    
    g : a vector of community labels

    Returns
    -------
    A dictionary with:
        (variable name)["B"] : estimated block connection probability matrix
        
        (variable name)["Phat"] : estimated probability matrix
        
        (variable name)["g"] : community labels
 
    Author(s)
    ----------
    Tianxi Li, Elizaveta Levina, Ji Zhu
    
    """
    
    n = A.shape[0]
    K = len(np.unique(g))
    B = np.zeros((K,K))
    M = np.zeros((n,K))
    for i in range(K):
        for j in range(i, K):
            if i != j:
                # in order to get each row and column needed, a bit of crazy
                # indexing was needed - could possibly be better some other way
                pre_mean = A[np.where(g == i),:][0]
                B[i,j] = np.mean(pre_mean[:,np.where(g == j)])
                B[j,i] = np.mean(pre_mean[:,np.where(g == j)])
            else:
                n_i = len(np.where(g==i)[0])
                pre_sum = A[np.where(g == i),:][0]
                B[i,i] = np.sum(pre_sum[:,np.where(g == i)])/(n_i**2 - n_i)
    
    # maybe a faster way ?
    for i in range(n):
        M[i,g[i]] = 1
    pP = np.matmul(M,B)
    P = np.matmul(pP,M.T)
    dic = {}
    dic["B"] = B
    dic["Phat"] = P
    dic["g"] = g
    return dic



def DCSBM_estimate(A,g):
    """
    Description
    ----------
    Estimates DCSBM model by given community labels
 
    Arguments
    ----------
    A : adjacency matrix
    
    g : vector of community labels for the nodes

    Returns
    -------
    A dictionary with:
        (variable name)["Phat"] : estimated probability matrix

        (variable name)["B"] : the B matrix with block connection probability, up to a scaling constant
                
        (variable name)["Psi"] : vector of of degree parameter theta, up to a scaling constant
        
    Author(s)
    ----------
    Tianxi Li, Elizaveta Levina, Ji Zhu
    
    """
    
    n = A.shape[0]
    K = len(np.unique(g))
    B = np.zeros((K,K))
    Theta = np.zeros((n,K))
    for i in range(K):
        for j in range(i,K):
            N_i = np.where(g==i)        
            psum = A[np.where(g == i),:][0]
            B[i,j] = np.sum(psum[:,np.where(g == j)]) + .001
            B[j,i] = np.sum(psum[:,np.where(g == j)]) + .001
        Theta[N_i,i] = 1
    Psi = np.sum(A, axis = 0)
    B_rowSums = np.sum(B, axis = 1)
    B_g = np.matmul(Theta, B_rowSums)
    Psi = (Psi/B_g).reshape(300,1)
    tmp_mat = Theta * Psi
    pP_hat = np.matmul(tmp_mat,B)
    P_hat = np.matmul(pP_hat,tmp_mat.T)
    dic = {}
    dic["Phat"] = P_hat
    dic["B"] = B
    dic["Psi"] = Psi.T
    dic["g"] = g
    return dic



def iter_SVD_core_fast_all(A, Kmax, tol = .00001, max_iter = 100, sparse = True,
                           init = None, verbose = False, tau = 0, p_sample = 1):
    """
    This function is used in ECV_block function
    """
    #?? make sparse?
    #if sparse == True:
        # possibly do this
        # A = 
        
    n = A.shape[0]
    cap = 1 #kappa*avg.p   
    A = np.where(np.isnan(A), 0, A)
    A = A/p_sample
    #svd.new <- svd(A,nu=K,nv=K)
    #print("begin SVD")
    
    svd_new = irlb(A, Kmax, maxit = max_iter) # might be a better SVD out there

    result = dict()
    
    for K in range(Kmax):
        # print(K) # not sure why this is here
        if K == 0:
            A_new = svd_new["S"][0] * np.matmul(np.array(svd_new["U"][:,0]).reshape(n,1), np.array(svd_new["V"][:,0]).reshape(n,1).T)
        else:
            A_new = A_new + svd_new["S"][K] * np.matmul(np.array(svd_new["U"][:,K]).reshape(n,1), np.array(svd_new["V"][:,K]).reshape(n,1).T)
        
        A_new_thr = A_new
        A_new_thr = np.where(A_new < (0 + tau), 0 + tau, A_new_thr)
        A_new_thr = np.where(A_new > cap, cap, A_new_thr)
        
        tmp_SVD = dict()
        tmp_SVD["u"] = svd_new["U"][:,range(K+1)]
        tmp_SVD["v"] = svd_new["V"][:,range(K+1)]
        tmp_SVD["d"] = svd_new["S"][range(K+1)]
        
        result[K] = {"iter": None, "SVD": tmp_SVD, "A": A_new, "err_seq": None, "A_thr": A_new_thr}
        
    return(result)
    


def ECV_block(A, max_K, cv = None, B = 3, holdout_p = 0.1, tau = 0, 
              kappa = None):
    """
    Description
    ----------
    Model selection by ECV for SBM and DCSBM. It can be used to select between the two models or given on model (either SBM or DCSBM) and select K.
        
    Arguments
    ----------
    A : adjacency matrix
    
    max_K : largest possible K for number of communities
    
    cv : cross validation fold. The default value is NULL. We recommend to use the argument B instead, doing indpendent sampling.
    
    B : number of replications
    
    holdout_p : testing set proportion
    
    tau : constant for numerical stability only. Not useful for current version.
    
    dc_est : estimation method for DCSBM. By defaulty (dc.est=2), the maximum likelihood is used. If dc.est=1, the method used by Chen and Lei (2016) is used, which is less stable according to our observation.
    
    kappa : constant for numerical stability only. Not useful for current version.

    Returns
    -------
    A dictionary with:
        (variable name)["impute_err"] : average validaiton imputation error
        
        (variable name)["l2"] : average validation L_2 loss under SBM
        
        (variable name)["dev"] : average validation binomial deviance loss under SBM
        
        (variable name)["auc"] : average validation AUC
        
        (variable name)["dc_l2"] : average validation L_2 loss under DCSBM
        
        (variable name)["dc_dev"] : average validation binomial deviance loss under DCSBM
        
        (variable name)["sse"] : average validation SSE
        
        (variable name)["l2_model"] : selected model by L_2 loss
        
        (variable name)["dev_model"] : selected model by binomial deviance loss
        
        (variable name)["l2_mat"] : cross-validation loss matrix for B replications
        
        (variable name)["dc_l2_mat"] : cross-validation loss matrix for B replications

    Author(s)
    ----------
    Tianxi Li, Elizaveta Levina, Ji Zhu
    
    """
    
    n = A.shape[0]
    edge_index = np.triu_indices(n, k = 1)
    edge_n = len(edge_index[0])
    holdout_index_list = list()
    
    if cv == None:
        holdout_n = math.floor(holdout_p * edge_n)
       
        for i in range(B):
            holdout_index_list.append(random.sample(range(edge_n), k = holdout_n))
            
    else:
        sample_index = random.sample(range(edge_n), k = edge_n)
        max_fold_num = np.ceil(edge_n/cv)
        fold_index = np.repeat(range(cv), max_fold_num)[edge_n-1]  
        cv_index = np.where(fold_index-(fold_index) == sample_index, fold_index, None)
        B = cv
        for i in range(B):
            holdout_index_list.append(np.where(cv_index == i)[0].tolist())
            
    def holdout_evaluation_fast_all(holdout_index, A = A, max_K = max_K, tau = tau,
                                p_sample = 1, kappa = kappa):

        n = A.shape[0]
        edge_index = np.triu_indices(n, k = 1)
        edge_n = len(edge_index[0])
        A_new = np.zeros(n**2).reshape(n,n)
        A_new[np.triu_indices_from(A_new,k = 1)] = A[edge_index]
        
        # may be better way to index here
        x = A_new[np.triu_indices_from(A_new,k = 1)]
        for i in range(len(holdout_index)):
            x[holdout_index[i]] = None
        A_new[np.triu_indices_from(A_new,k = 1)] = x
    
        A_new = A_new + A_new.T
        degrees = np.nansum(A_new, axis = 1)
        no_edge = 0
        no_edge = np.sum(degrees == 0)  
        
        Omega = np.isnan(A_new)
        non_miss = ~np.isnan(A_new)
        #A.new[non.miss] <- A.new[non.miss] + 0.5
    
        SVD_result = iter_SVD_core_fast_all(A_new, max_K, p_sample = p_sample)
        
        dc_block_sq_err = [0] * max_K
        dc_loglike = [0] * max_K
        roc_auc = [0] * max_K
        bin_dev = [0] * max_K
        block_sq_err = [0] * max_K
        impute_sq_err = [0] * max_K
        loglike = [0] * max_K
    
        for k in range(max_K):
            tmp_est = SVD_result[k]
            A_approx = tmp_est["A_thr"]
            impute_sq_err[k] = np.sum((A_approx[Omega] - A[Omega])**2)
            
            response = list()
            upper_A = A[np.triu_indices_from(A, k = 1)]
            for i in range(len(holdout_index)):
                response.append(upper_A[holdout_index[i]])
            
            predictors = list()
            upper_A_approx = A_approx[np.triu_indices_from(A_approx, k = 1)]
            for i in range(len(holdout_index)):
                predictors.append(upper_A_approx[holdout_index[i]])    
        
            print("AUC calculation")
            fpr, tpr, threshold = metrics.roc_curve(response, predictors, pos_label=1)
            roc_auc[k] = metrics.auc(fpr, tpr)
            trunc_predictors = np.array(predictors) # changing to an array to compare values
            trunc_predictors[np.array(predictors) > (1-1e-6)] = 1-1e-6
            trunc_predictors[np.array(predictors) < (1e-6)] = 1e-6
            bin_dev[k] = np.sum((np.array(response) - trunc_predictors)**2) 
    
            if k == 0:
                pb = (np.nansum(A_new) + 1)/ (np.sum(~np.isnan(A_new)) - np.sum(~np.isnan(np.diag(A_new))) + 1)
                if pb < 1e-6:
                    pb = 1e-6
                if pb > 1-1e-6:
                    pb = 1-1e-6
                A_Omega = A[Omega]
                block_sq_err[k] = np.sum((pb - A_Omega)**2)
                loglike[k] = -np.sum(A_Omega * np.log(pb)) - np.sum((1 - A_Omega) * np.log(1-pb))
    
            print("SBM calculation")
            start = timeit.default_timer()
            
            if k == 0:
                U_approx = np.array(tmp_est["SVD"]["v"]).reshape(len(tmp_est["SVD"]["v"]), k+1)
            else:
                U_approx = tmp_est["SVD"]["v"][:,range(k+1)]
                
                if tau > 0:
                    A_approx = A_approx + (tau * np.mean(np.sum(A_approx, axis = 0))/n)
                    d_approx = np.sum(A_approx, axis = 0)
                    preL_approx = np.matmul(np.diag(1/np.sqrt(d_approx)), A_approx)
                    L_approx = np.matmul(preL_approx, np.diag(1/np.sqrt(d_approx)))
                    A_approx_svd = irlb(L_approx, k+1, maxit = 1000)
                    U_approx = A_approx_svd["V"][:,range(k+1)] 
                
            km = KMeans(n_clusters = k + 1, max_iter = 30).fit(U_approx)
            B = np.zeros((k+1,k+1))
            Theta = np.zeros((n, k+1))        
    
            for i in range(k+1):
                for j in range(i, k+1):
                    N_i = np.where(km.labels_ == i)
                    N_j = np.where(km.labels_ == j)
                    if i != j:
                        B[i,j] = (np.nansum(A_new[N_i[0][:, None], N_j[0][None, :]]) + 1)/ (np.sum(~np.isnan(A_new[N_i[0][:, None], N_j[0][None, :]]))+1) # i believe this is the same indexing, but was having trouble figuring it out
                        B[j,i] = (np.nansum(A_new[N_i[0][:, None], N_j[0][None, :]]) + 1)/ (np.sum(~np.isnan(A_new[N_i[0][:, None], N_j[0][None, :]]))+1)
                    else:
                        B[i,j] = (np.nansum(A_new[N_i[0][:, None], N_j[0][None, :]]) + 1)/(np.sum(~np.isnan(A_new[N_i[0][:, None], N_j[0][None, :]])) - np.sum(~np.isnan(np.diag(A_new[N_i[0][:, None], N_j[0][None, :]])))+1)
                        B[j,i] = (np.nansum(A_new[N_i[0][:, None], N_j[0][None, :]]) + 1)/(np.sum(~np.isnan(A_new[N_i[0][:, None], N_j[0][None, :]])) - np.sum(~np.isnan(np.diag(A_new[N_i[0][:, None], N_j[0][None, :]])))+1)
                Theta[N_i,i] = 1
            
            preP_hat = np.matmul(Theta, B)
            P_hat = np.matmul(preP_hat, Theta.T)
            np.fill_diagonal(P_hat,0)
            block_sq_err[k] = np.sum((P_hat[Omega]-A[Omega])**2)
            P_hat_Omega = P_hat[Omega]
            A_Omega = A[Omega]        
            P_hat_Omega[P_hat_Omega < 1e-6] = 1e-6
            P_hat_Omega[P_hat_Omega > (1-1e-6)] = 1-1e-6
            loglike[k] = -np.sum(A_Omega*np.log(P_hat_Omega)) - np.sum((1-A_Omega)* np.log(1-P_hat_Omega))
    
            stop = timeit.default_timer()
            print('Time: ', stop - start) 
            
            #### Degree correct model
            V = U_approx
            print("DCSBM calculation")
            start = timeit.default_timer()
            if k == 0:
                V_norms = np.abs(V.reshape(1, n * (k+1))[0])
            else:
                def sq_sum(x):
                    return(np.sqrt(np.sum(x**2))) # made this to use the apply func
                V_norms = np.apply_along_axis(sq_sum, 1, V)
            
            iso_index = np.where(V_norms == 0)
            Psi = V_norms
            Psi = Psi / np.max(V_norms)
            inv_V_norms = 1/V_norms
            inv_V_norms[iso_index] = 1 # this should work but indexing may be different here
    
            V_normalized = np.matmul(np.diag(inv_V_norms), V)
    
            if k == 0:
                B = np.nansum(A_new) + 0.01
                partial_d = np.nansum(A_new, axis = 0)
                partial_gd = B
                phi = [0] * n
                B_g = partial_gd
                phi = partial_d/B_g
                B = B/p_sample
                P_hat = ((np.array([B] * (n*n)).reshape(n,n) * phi).T * phi).T
                np.fill_diagonal(P_hat,0)
                
                dc_block_sq_err[k] = np.sum((pb - A[Omega])**2)
                P_hat_Omega = P_hat[Omega]
                A_Omega = A[Omega]
                P_hat_Omega[P_hat_Omega < 1e-6] = 1e-6        
                P_hat_Omega[P_hat_Omega > (1-1e-6)] = 1-1e-6
                dc_loglike[k] = -np.sum(A_Omega * np.log(P_hat_Omega)) - np.sum((1 - A_Omega)*np.log(1-P_hat_Omega))          
           
            else:
                km = KMeans(n_clusters = k + 1, max_iter = 30).fit(V_normalized)
                B = np.zeros((k+1,k+1))
                Theta = np.zeros((n, k+1))
                for i in range(k+1):
                    for j in range(k+1): 
                        N_i = np.where(km.labels_ == i)
                        N_j = np.where(km.labels_ == j)
                        B[i,j] = (np.nansum(A_new[N_i[0][:, None], N_j[0][None, :]]) + 0.01)
                    Theta[N_i,i] = 1
                    
                partial_d = np.nansum(A_new, axis = 0)
                partial_gd = np.sum(B, axis = 0)
                phi = [0] * n
                B_g = np.matmul(Theta, partial_gd)
                phi = partial_d/B_g
                B = B/p_sample
                tmp_int_mat = Theta * phi[:, None]
                preP_hat = np.matmul(tmp_int_mat, B)
                P_hat = np.matmul(preP_hat, tmp_int_mat.T)
                np.fill_diagonal(P_hat,0)
            
                dc_block_sq_err[k] = np.sum((P_hat[Omega]-A[Omega])**2)
                P_hat_Omega = P_hat[Omega]
                A_Omega = A[Omega]                
                P_hat_Omega[P_hat_Omega < 1e-6] = 1e-6        
                P_hat_Omega[P_hat_Omega > (1-1e-6)] = 1-1e-6
                dc_loglike[k] = -np.sum(A_Omega * np.log(P_hat_Omega)) - np.sum((1 - A_Omega)*np.log(1-P_hat_Omega))          
            stop = timeit.default_timer()
            print('Time: ', stop - start) 
            
        dic = {}
        dic["impute_sq_err"] = impute_sq_err
        dic["block_sq_err"] = block_sq_err
        dic["loglike"] = loglike
        dic["roc_auc"] = roc_auc
        dic["no_edge"] = no_edge
        dic["dc_block_sq_err"] = dc_block_sq_err
        dic["dc_loglike"] = dc_loglike
        dic["bin_dev"] = bin_dev
        
        return dic
      
    def my_lapply(lst):
        dic = {}
        j = 0
        for i in lst:
            dic[j] = holdout_evaluation_fast_all(i, max_K = max_K, A = A, p_sample = 1 - holdout_p)
            j += 1
        return dic
                   
    result = my_lapply(holdout_index_list)
    
    
    dc_block_err_mat = np.zeros((B, max_K))
    dc_loglike_mat = np.zeros((B, max_K))
    bin_dev_mat = np.zeros((B, max_K))
    roc_auc_mat = np.zeros((B, max_K))
    impute_err_mat = np.zeros((B, max_K))
    block_err_mat = np.zeros((B, max_K))
    loglike_mat = np.zeros((B, max_K))
    no_edge_seq = [0] * B
    
    for b in range(0,B):
        impute_err_mat[b,] = result[b]["impute_sq_err"]
        block_err_mat[b,] = result[b]["block_sq_err"]
        loglike_mat[b,] = result[b]["loglike"]
        roc_auc_mat[b,] = result[b]["roc_auc"]
        bin_dev_mat[b,] = result[b]["bin_dev"]
        no_edge_seq[b] = result[b]["no_edge"]
        dc_block_err_mat[b,] = result[b]["dc_block_sq_err"]
        dc_loglike_mat[b,] = result[b]["dc_loglike"]
        
    output = {}
    output["impute_err"] = np.mean(impute_err_mat,axis = 0)
    output["l2"] = np.mean(block_err_mat, axis = 0)
    output["dev"] = np.sum(loglike_mat, axis = 0)
    output["auc"] = np.mean(roc_auc_mat, axis = 0)       
    output["dc_l2"] = np.mean(dc_block_err_mat, axis = 0)
    output["dc_dev"] = np.sum(dc_loglike_mat, axis = 0)
    output["sse"] = np.mean(impute_err_mat, axis = 0)
    output["auc_mat"] = roc_auc_mat
    output["dev_mat"] = loglike_mat
    output["l2_mat"] = block_err_mat
    output["SSE_mat"] = impute_err_mat
    output["dc_dev_mat"] = dc_loglike_mat
    output["dc_l2_mat"] = dc_block_err_mat
       
    if np.min(output["dev"]) > np.min(output["dc_dev"]):
        #?? should i change it to 1-6 or leave index as 0-5?
        dev_model = "DCSBM-" + str(list(output["dc_dev"]).index(min(list(output["dc_dev"]))))
    else:
        dev_model = "SBM-" + str(list(output["dev"]).index(min(list(output["dev"]))))
    
    if np.min(output["l2"]) > np.min(output["dc_l2"]):
        l2_model = "DCSBM-" + str(list(output["dc_l2"]).index(min(list(output["dc_l2"]))))
    else:
        l2_model = "SBM-" + str(list(output["l2"]).index(min(list(output["l2"]))))

    output["l2_model"] = l2_model
    output["dev_model"] = dev_model
    
    return output



def RDPG_Gen(n, K, directed = True, avg_d = None):
    """
    Description
    ----------
    Generates random networks from random dot product graph model  
      
    Arguments
    ----------
    n : size of the network
    
    K : dimension of latent space
    
    directed : whether the network is directed or not
    
    avg_d : average node degree of the network (in expectation)

    Returns
    -------
    A dictionary with:
        (variable name)["A"] : the adjacency matrix
        
        (variable name)["P"] : the probability matrix

    Author(s)
    ----------
    Tianxi Li, Elizaveta Levina, Ji Zhu
    
    """
    
    if directed == True:
        return DirectedRDPG(n, K, avg_d)
    else:
        return UndirectedRDPG(n, K, avg_d)



def UndirectedRDPG(n, K, avg_d = None):
    """
    This function is used in the RDPG_Gen function
    """
    Z1 = np.array(np.abs(np.random.uniform(size = n*K))).reshape(n,K)
    S = np.matmul(Z1, Z1.T)
    P = S/np.max(S)
    if avg_d != None:
        P = (P * avg_d)/(np.mean(P.sum(axis = 1)))
    upper_index = np.triu_indices(n, k = 1)
    upper_p = P[upper_index]
    upper_u = np.random.uniform(size = len(upper_p))
    upper_A = np.where(upper_u < upper_p, 1, 0)
    
    A = np.zeros((n, n), dtype=np.int32)
    A[upper_index] = upper_A
    A = A + A.T
    dic = {}
    dic["A"] = A
    dic["P"] = P
    return dic



def DirectedRDPG(n, K, avg_d = None):
    """
    This function is used in the RDPG_Gen function
    """
    Z1 = np.array(np.abs(np.random.uniform(size = n*K))).reshape(n,K)
    Z2 = np.array(np.abs(np.random.uniform(size = n*K))).reshape(n,K)
    S = np.matmul(Z1, Z2.T)
    P = S/np.max(S)
    if avg_d != None:
        P = (P * avg_d)/(np.mean(P.sum(axis = 1)))
    A = np.zeros((n,n))
    R = np.array(np.random.uniform(size = n**2)).reshape(n,n)
    A[R<P] = 1
    dic = {}
    dic["A"] = A
    dic["P"] = P
    return dic



def ECV_Rank(A, max_K, B = 3, holdout_p = 0.1, weighted = True, mode = "directed"):
    """
    Description
    ----------
    Estimates the optimal low rank model for a network 
     
    Arguments
    ----------
    A : adjacency matrix
    
    max_K : maximum possible rank to check
    
    B : number of replications in ECV
    
    holdout_p : test set proportion
    
    weighted : whether the network is weighted. If TRUE, only sum of squared errors are computed. If FALSE, then treat the network as binary and AUC will be computed along with SSE.
    
    mode : Selectign the mode of "directed" or "undirected" for cross-validation.

    Returns
    -------
    A dictionary with:
        (variable name)["sse_rank"] : rank selection by SSE loss
        
        (variable name)["auc_rank"] : rank selection by AUC loss
        
        (variable name)["auc"] : auc sequence for each rank candidate
        
        (variable name)["sse"] : sse sequence for each rank candidate

    Author(s)
    ----------
    Tianxi Li, Elizaveta Levina, Ji Zhu
    
    """
    if mode == "directed":
        return(ECV_directed_Rank(A=A, max_K = max_K, B=B, holdout_p = holdout_p, weighted = weighted))
    else:
        return(ECV_undirected_Rank(A=A, max_K = max_K, B=B, holdout_p = holdout_p, weighted = weighted))
    
        
    
def ECV_directed_Rank(A, max_K, B = 3, holdout_p = 0.1, weighted = True):
    """
    This function is used in the ECV_Rank function
    """
    n = A.shape[0]
    edge_index = np.zeros((n,n))
    edge_n = len(edge_index[0])**2 

    holdout_index_list = list()
    holdout_n = math.floor(holdout_p * edge_n)
    for i in range(B):
        holdout_index_list.append(random.sample(range(edge_n), k = holdout_n))
        
    def my_lapply(lst):
        dic = {}
        j = 0
        for i in lst:
            dic[j] = missing_directed_rank_fast_all(i, max_K = max_K, A = A, p_sample = 1 - holdout_p, weighted = weighted)
            j += 1
        return dic
                   
    result = my_lapply(holdout_index_list)
    sse_mat = np.zeros((B, max_K))
    roc_auc_mat = np.zeros((B, max_K))
    
    for b in range(B):
        roc_auc_mat[b,] = result[b]["roc_auc"]
        sse_mat[b,] = result[b]["sse"]
    
    if weighted == False:
        auc_seq = np.mean(roc_auc_mat, axis = 0)
    else:
        auc_seq = [None] * max_K
        
    sse_seq = np.mean(sse_mat, axis = 0)
    
    dic = dict()
    dic["sse_rank"] = list(sse_seq).index(min(list(sse_seq))) + 1
    if np.isnan(auc_seq[0]): # this is just checking if it is None so that the index function will work
        dic["auc_rank"] = None
    else:
        dic["auc_rank"] = list(auc_seq).index(max(list(auc_seq))) + 1
    dic["auc"] = auc_seq
    dic["sse"] = sse_seq
    
    return dic
    
    
    
def missing_directed_rank_fast_all(holdout_index, A, max_K, p_sample = None, weighted = True):
    """
    This function is used in the ECV_Rank function
    """
    n = A.shape[0]
    A_new = deepcopy(A)
    x = A_new.reshape(n**2,1)
    for i in range(len(holdout_index)):
        x[holdout_index[i]] = 5 # would not let me directly change to np.nan
    x = np.where(x == 5, np.nan, x) # so used np.where to change after loop
    A_new = x.reshape(n,n) # made sure this was by.row so this indexing works!
    
    if p_sample == None:
        p_sample = 1 - (len(holdout_index)/n**2)
    degrees = np.nansum(A_new, axis = 0)
    no_edge = 0
    no_edge = np.sum(degrees == 0)        

    Omega = np.isnan(A_new)
    imputed_A = dict()
    roc_auc = [0] * max_K
    sse = [0] * max_K
    
    SVD_result = iter_SVD_core_fast_all(A_new, max_K, p_sample = p_sample)
    
    for k in range(max_K):
        tmp_est = SVD_result[k]
        A_approx = tmp_est["A_thr"]
        response = A[Omega]
        predictors = A_approx[Omega]
        if weighted == False:
            fpr, tpr, threshold = metrics.roc_curve(response, predictors, pos_label=1)
            roc_auc[k] = metrics.auc(fpr, tpr)
            
        sse[k] = np.mean((response - predictors)**2)
        imputed_A[k] = A_approx
        
    dic = {}
    dic["imputed_A"] = imputed_A
    dic["Omega"] = Omega
    dic["sse"] = sse
    dic["roc_auc"] = roc_auc
    
    return dic
        

   
def ECV_undirected_Rank(A, max_K, B = 3, holdout_p = 0.1, weighted = True):
    """
    This function is used in the ECV_Rank function
    """
    n = A.shape[0]
    edge_index = np.triu_indices(n, k = 1)
    edge_n = len(edge_index[0])   
    # edge.index <- 1:n^2
    # edge.n <- length(edge.index)
        
    holdout_index_list = list()
    holdout_n = math.floor(holdout_p * edge_n)
    for i in range(B):
        holdout_index_list.append(random.sample(range(edge_n), k = holdout_n))
    
    def my_lapply(lst):
        dic = {}
        j = 0
        for i in lst:
            dic[j] = missing_undirected_rank_fast_all(i, max_K = max_K, A = A, p_sample = 1 - holdout_p, weighted = weighted)
            j += 1
        return dic
                   
    result = my_lapply(holdout_index_list)
    sse_mat = np.zeros((B, max_K))
    roc_auc_mat = np.zeros((B, max_K))
    
    for b in range(B):
        roc_auc_mat[b,] = result[b]["roc_auc"]
        sse_mat[b,] = result[b]["sse"]
    
    if weighted == False:
        auc_seq = np.mean(roc_auc_mat, axis = 0)
    else:
        auc_seq = [None] * max_K
        
    sse_seq = np.mean(sse_mat, axis = 0)
    
    dic = dict()
    dic["sse_rank"] = list(sse_seq).index(min(list(sse_seq))) + 1
    if auc_seq[0] == None: # this is just checking if it is None so that the index function will work
        dic["auc_rank"] = None
    else:
        dic["auc_rank"] = list(auc_seq).index(max(list(auc_seq))) + 1
    dic["auc"] = auc_seq
    dic["sse"] = sse_seq
    
    return dic
        

    
def missing_undirected_rank_fast_all(holdout_index, A, max_K, p_sample = 1, weighted = True):
    """
    This function is used in the ECV_Rank function
    """
    n = A.shape[0]
    edge_index = np.triu_indices(n, k = 1)
    edge_n = len(edge_index[0])
    A_new = np.zeros(n**2).reshape(n,n)
    A_new[np.triu_indices_from(A_new,k = 1)] = A[edge_index]
    
    # may be better way to index here
    x = A_new[np.triu_indices_from(A_new,k = 1)]
    for i in range(len(holdout_index)):
        x[holdout_index[i]] = None
    A_new[np.triu_indices_from(A_new,k = 1)] = x

    A_new = A_new + A_new.T
    np.fill_diagonal(A_new, np.diag(A))
    degrees = np.nansum(A_new, axis = 0)
    no_edge = 0
    no_edge = np.sum(degrees == 0)  
    
    Omega = np.isnan(A_new)
    imputed_A = dict()
    roc_auc = [0] * max_K
    sse = [0] * max_K
    
    SVD_result = iter_SVD_core_fast_all(A_new, max_K, p_sample = p_sample)
    
    for k in range(max_K):
        tmp_est = SVD_result[k]
        A_approx = tmp_est["A"]
        response = A[Omega]
        predictors = A_approx[Omega]
        if weighted == False:
            fpr, tpr, threshold = metrics.roc_curve(response, predictors, pos_label=1)
            roc_auc[k] = metrics.auc(fpr, tpr)
            
        sse[k] = np.mean((response - predictors)**2)
        imputed_A[k] = A_approx
    
    dic = {}
    dic["imputed_A"] = imputed_A
    dic["Omega"] = Omega
    dic["sse"] = sse
    dic["roc_auc"] = roc_auc
    
    return dic


