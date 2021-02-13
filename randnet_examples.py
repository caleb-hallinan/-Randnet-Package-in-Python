#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 15:18:28 2020

@author: calebhallinan
"""

### Randnet script testing example ###


## Imporintng randnet
import randnet

## To look at documentation of each function, use:
# help(randnet.(function name))
# OR
# type the function out and click on documentation


## BlockModelGen example ##
# Generates networks from degree corrected stochastic block model, with various options for node degree distribution
dt = randnet.BlockModelGen(30, 300, K = 3, beta = 0.2, rho = 0.9, simple = False, power = True)

# A - the generated network adjacency matrix
dt["A"]
# g - community membership
dt["g"]
# P - probability matrix of the network
dt["P"]
# theta - node degree parameter
dt["theta"]

## reg_SP example ##
# Community detection by regularized spectral clustering
sc = randnet.reg_SP(dt["A"], K = 3, lap = True)

# cluster - cluster labels
sc["cluster"]
# sc - the loss of Kmeans algorithm
sc["loss"]

## reg_SSP example ##
# Community detection by regularized spherical spectral clustering
ssc = randnet.reg_SSP(dt["A"], K = 3, lap = True)

# cluster - cluster labels
ssc["cluster"]
# loss - the loss of Kmeans algorithm
ssc["loss"]

## SBM_estimate example ##
# Estimates SBM parameters given community labels
sbm = randnet.SBM_estimate(dt["A"], sc["cluster"])

# B - estimated block connection probability matrix
sbm["B"]
# Phat - estimated probability matrix
sbm["Phat"]
# g - community labels
sbm["g"]

## DCSBM_estimate exampel ##
# Estimates DCSBM model by given community labels
est = randnet.DCSBM_estimate(dt["A"], ssc["cluster"])

# Phat - estimated probability matrix
est["Phat"]
# B - the B matrix with block connection probability, up to a scaling constant
est["B"]
# Psi - vector of of degree parameter theta, up to a scaling constant
est["Psi"]

# ECV_block example ##
# Model selection by ECV for SBM and DCSBM. It can be used to select between the two models or given on model (either SBM or DCSBM) and select K.
ecv = randnet.ECV_block(dt["A"], 6, B = 3)

#impute_err - average validaiton imputation error
ecv["impute_err"]
#l2 - average validation L_2 loss under SBM
ecv["l2"]
#dev - average validation binomial deviance loss under SBM
ecv["dev"]
#auc - average validation AUC
ecv["auc"]
#dc_l2 - average validation L_2 loss under DCSBM
ecv["dc_l2"]
#dc_dev - average validation binomial deviance loss under DCSBM
ecv["dc_dev"]
#sse - average validation SSE
ecv["sse"]
#l2_model - selected model by L_2 loss
ecv["l2_model"]
#dev_model - selected model by binomial deviance loss
ecv["dev_model"]
#l2_mat, dc_l2_mat,... - cross-validation loss matrix for B replications
ecv["l2_mat"]
ecv["dc_l2_mat"]

## ECV_Rank example ##
# Estimates the optimal low rank model for a network
ecv_rank = randnet.ECV_Rank(dt["A"], 6, weighted = False, mode = "undirected")

#sse_rank - rank selection by SSE loss
ecv_rank["sse_rank"]
#auc_rank - rank selection by AUC loss
ecv_rank["auc_rank"]
#auc - auc sequence for each rank candidate
ecv_rank["auc"]
#sse - sse sequence for each rank candidate
ecv_rank["sse"]

## RDPG_Gen example ##
# Generates random networks from random dot product graph model
rdpg = randnet.RDPG_Gen(n = 600, K = 2, directed = True)

# A - the adjacency matrix
rdpg["A"]
# P - the probability matrix
rdpg["P"]

