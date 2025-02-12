#%% Start with loading required packages

import os
import warnings

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

import jax.numpy as jnp
from jax import lax, random
from jax.scipy.special import expit

import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import effective_sample_size
from numpyro.infer import MCMC, NUTS, Predictive, init_to_mean, init_to_median

az.style.use("arviz-darkgrid")
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)

seed = random.PRNGKey(5)

#%% Note for working with 'real' experimental data, we have set up and defined our model and data such that:
# Re-defined original dataframe as all rounds but the Baseline and instead 
# re-formulate the output variable as change in y in relation to that previous round

#%% ------------------------------------
# Create a Dataframe and Dictonary only containing the 'structure' of later real data
# to use for storing the synthetic data and ensure correct structure
#---------------------------------------
n_r = 3
N = 100
#%%
dat_list_onlyIDs = dict(
    r=jnp.array([*jnp.tile(0, N),*jnp.tile(1, N),*jnp.tile(2, N)]),
    treat=jnp.tile(jnp.array([*jnp.tile(0, int(N/2)),*jnp.tile(1, int(N/2))]), n_r),
)
#%% ------------------------------------------------------------------------------
# Partially pooled model with varying sigma and roundeff per treatment group
# --------------------------------------------------------------------------------
#includes meta-level where sigma priors depend on the treatment group

def model_partpool(r, treat, delta_PSR_share=None, delta_SS_share=None): 
    
    # Define and set prior for the round effect- beta 
    roundeff = numpyro.sample("roundeff", dist.Normal(0, 0.3), sample_shape=(2,n_r)) #per technology and round

    # Define a prior for the average variance across rounds (pooling)
    sigma_bar = numpyro.sample('sigma_bar', dist.TruncatedNormal(loc=1, scale=0.2, low=0))
    # Use this average variance to set priors for the round-specific variance terms
    sigma_round = numpyro.sample("sigma_round", dist.Exponential(sigma_bar), sample_shape=(2,n_r)) #per round and treat

    # Define and set prior for the treatment beta
    btreat = numpyro.sample("btreat", dist.Normal(0, 0.3).expand([2])) #per tech

    # Define the means for each technology's posterior distribution to sample from 
    mu_delta_PSR = roundeff[0,r] + btreat[0]*treat
    mu_delta_SS = roundeff[1,r] + btreat[1]*treat

    # Define the round- and treatment- specific variance terms for the posterior distributions
    sigma = sigma_round[treat, r]

    # Sample from Truncated normal distributions with above-defined means and standard deviations to obtain posteriors
    delta_PSR_share = numpyro.sample('delta_PSR_share', dist.TruncatedNormal(loc=mu_delta_PSR, scale=sigma,low=-1, high=1), obs=delta_PSR_share)
    delta_SS_share = numpyro.sample('delta_SS_share', dist.TruncatedNormal(loc=mu_delta_SS, scale=sigma,low=-1, high=1), obs=delta_SS_share)

#%% ----------------------------------------------------------------------------
# Conditioned Data-Generating Process Using the Partial Pooling Model
# ------------------------------------------------------------------------------
#%% --------------------------------
# First Conditioning : True H1
# The roundeff in r=0 (when moving from r0 in r1 of the experiment) is non-zero
#-----------------------------------
coef_trueH1_partpool = {
    'sigma_round':jnp.array([[.2, .2 ,.2],[.2 , .2, .2 ]]),
    'roundeff':jnp.array([[-0.2,0,0],[-0.2,0,0]]),
    'btreat':jnp.array([0.,0.],)
    }
#%% Draw a synthetic Dataset following these conditioned parameter values
partpool_condModel_H1 = numpyro.handlers.condition(model_partpool, data=coef_trueH1_partpool)
prior_predictive_Partpool_Cond_H1 = Predictive(partpool_condModel_H1, num_samples=1)
#%% Define the structure of the synthetic Dataset so that it follows our real structure
prior_samples_Partpool_Cond_H1 = prior_predictive_Partpool_Cond_H1(seed, **dat_list_onlyIDs)
#%% Check whether the conditioned values were correctly set
print('roundeff', prior_samples_Partpool_Cond_H1['roundeff'])
print('sigma_round', prior_samples_Partpool_Cond_H1['sigma_round'][0,0])
print('btreat', prior_samples_Partpool_Cond_H1['btreat'])
#%% Save the conditioned syntethic Sample (Dataset) into a Dataframe to make plotting easier
dfPrior = pd.DataFrame(columns=[ 'treat', 'r', 'delta_PSR_share', 'delta_SS_share','delta_BB_share'])
dfPrior['r'] = dat_list_onlyIDs.get('r')
dfPrior['treat'] = dat_list_onlyIDs.get('treat')
dfPrior
#%%
dfPrior['delta_PSR_share'] = prior_samples_Partpool_Cond_H1['delta_PSR_share'][0,:]
dfPrior['delta_SS_share'] = prior_samples_Partpool_Cond_H1['delta_SS_share'][0,:]
dfPrior['delta_BB_share'] = -(dfPrior['delta_PSR_share'] + dfPrior['delta_SS_share'])
dfPrior
#%% Plot sampled PSR for the different rounds
#Figure 1, first row, left graphic
fig,ax = plt.subplots(layout='constrained')
ax.hist(dfPrior.loc[((dfPrior['r']==1)),'delta_PSR_share'].values.flatten(),bins=50, label='r1',alpha = 0.5, color='red')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==1)),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='red')
ax.hist(dfPrior.loc[((dfPrior['r']==0)),'delta_PSR_share'].values.flatten(),bins=50, label='r0',alpha = 0.3, color='blue')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==0)),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='blue')
ax.set(xlabel='Value of delta_WR', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')
#%%
#Figure 1, first row, right graphic
fig,ax = plt.subplots(layout='constrained')
ax.hist(dfPrior.loc[(dfPrior['r']==2) ,'delta_SS_share'].values.flatten(),bins=50, label='r2',alpha = 0.5, color='green')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==2)),'delta_SS_share'].values.flatten(), alpha = 0.5, color='green')
ax.hist(dfPrior.loc[((dfPrior['r']==1)),'delta_SS_share'].values.flatten(),bins=50, label='r1',alpha = 0.5, color='red')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==1)),'delta_SS_share'].values.flatten(), alpha = 0.5, color='red')
ax.set(xlabel='Value of delta_SS', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')
#%% -----------
# Run a MCMC-Analysis with this conditioned synthetic Data
# - test whether the 'true' conditioned parameter values can be obtained here to ensure model functionality
#--------------
model_dummy_mcmc_H1_partpool = MCMC(NUTS(model_partpool, init_strategy = init_to_median), num_warmup=1000, num_samples=1000, num_chains=4)
model_dummy_mcmc_H1_partpool.run(seed, r=dat_list_onlyIDs.get('r'),
    treat=dat_list_onlyIDs.get('treat'),
    delta_PSR_share = prior_samples_Partpool_Cond_H1['delta_PSR_share'][0,:], 
    delta_SS_share = prior_samples_Partpool_Cond_H1['delta_SS_share'][0,:])

mcmc_summary_dummy_H1_partpool = model_dummy_mcmc_H1_partpool.print_summary(0.89)
#%% Draw samples from the posterior distributions obtained in the MCMC to use for visualization
dummy_sample_H1_partpool = model_dummy_mcmc_H1_partpool.get_samples(group_by_chain=True)
#%% Posterior plotting for MCMC sample conditioned following H1
# Structure of the sample array: First dimension: chain (0-4), second dimension: sample draw (always include all), 
# third demension: technology (WR or SS), last dimesnion: round
# So: [chain, : , tech, round]
#%% sigma plotting 
# Figure 1, second row, left graphic:
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][0,:,0,0].flatten(), label='$\sigma_{control,0}$', color='darkgreen', alpha=0.7)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][1,:,0,0].flatten(),  color='darkgreen', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][2,:,0,0].flatten(),  color='darkgreen', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][3,:,0,0].flatten(),  color='darkgreen', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][0,:,0,1].flatten(), label='$\sigma_{control,1}$', color='darkred', alpha=0.7)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][1,:,0,1].flatten(),  color='darkred', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][2,:,0,1].flatten(),  color='darkred', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][3,:,0,1].flatten(),  color='darkred', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][0,:,0,2].flatten(), label='$\sigma_{control,2}$', color='darkblue', alpha=0.7)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][1,:,0,2].flatten(),  color='darkblue', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][2,:,0,2].flatten(),  color='darkblue', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][3,:,0,2].flatten(),  color='darkblue', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][0,:,1,0].flatten(), label='$\sigma_{treat,0}$', color='lightgreen', alpha=0.7)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][1,:,1,0].flatten(),  color='lightgreen', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][2,:,1,0].flatten(),  color='lightgreen', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][3,:,1,0].flatten(),  color='lightgreen', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][0,:,1,1].flatten(), label='$\sigma_{treat,1}$', color='darkorange', alpha=0.7)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][1,:,1,1].flatten(),  color='darkorange', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][2,:,1,1].flatten(),  color='darkorange', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][3,:,1,1].flatten(),  color='darkorange', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][0,:,1,2].flatten(), label='$\sigma_{treat,2}$', color='lightblue', alpha=0.7)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][1,:,1,2].flatten(),  color='lightblue', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][2,:,1,2].flatten(),  color='lightblue', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['sigma_round'][3,:,1,2].flatten(),  color='lightblue', alpha=0.5)
ax.axvline(x=0.2, color='black', ls='--', lw=2, label='Cond.Value')
ax.set_xlabel('$\sigma_{treat,r}$')
ax.set_ylabel('Density')
ax.set_xlim([0.12,0.35])
ax.legend(loc='upper right')
#%%
#Figure 1, second row, right graphic
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][1,:,0,0].flatten(), label='$\\theta_{WR,0}$', color='green', alpha=0.7)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][2,:,0,0].flatten(),  color='green', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][3,:,0,0].flatten(),  color='green', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][0,:,0,0].flatten(),  color='green', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][1,:,0,1].flatten(), label='$\\theta_{WR,1}$', color='red', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][2,:,0,1].flatten(),  color='red', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][3,:,0,1].flatten(),  color='red', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][0,:,0,1].flatten(),  color='red', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][1,:,0,2].flatten(), label='$\\theta_{WR,2}$', color='blue', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][2,:,0,2].flatten(),  color='blue', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][3,:,0,2].flatten(),  color='blue', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][0,:,0,2].flatten(),  color='blue', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][0,:,1,0].flatten(), label='$\\theta_{SS,0}$', color='purple', alpha=0.7)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][1,:,1,0].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][2,:,1,0].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][3,:,1,0].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][0,:,1,1].flatten(), label='$\\theta_{SS,1}$', color='orange', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][1,:,1,1].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][2,:,1,1].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][3,:,1,1].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][0,:,1,2].flatten(), label='$\\theta_{SS,2}$', color='grey', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][1,:,1,2].flatten(),  color='grey', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][2,:,1,2].flatten(),  color='grey', alpha=0.5)
sb.kdeplot(dummy_sample_H1_partpool['roundeff'][3,:,1,2].flatten(),  color='grey', alpha=0.5)
ax.axvline(x=-0.2, color='black', ls='--', lw=2, label='Cond.Value')
ax.axvline(x=0, color='black', ls='--', lw=2)
#ax.set_title('Posterior Density for the $roundeffect$ Parameter')    
ax.set_xlabel('$\\theta_{r,tech}$')
ax.set_ylabel('Density')
ax.set_xlim([-0.3,0.25])
ax.legend(loc='upper right')

#%% -------------------------------- 
# Second : True H2 - so the variance decreases over rounds, but faster for the control!
# ----------------------------------
coef_trueH2_partpool = {
    'sigma_round':jnp.array([[.2, .15 ,.1],[.2 , .17, .12 ]]),
    'roundeff':jnp.array([[0,0,0],[0,0,0]]),
    'btreat':jnp.array([0.,0.],)
    }
#%% Generate synthetic data which proves this 
partpool_condModel_H2 = numpyro.handlers.condition(model_partpool, data=coef_trueH2_partpool)
prior_predictive_Partpool_Cond_H2 = Predictive(partpool_condModel_H2, num_samples=1)
#%%
prior_samples_Partpool_Cond_H2 = prior_predictive_Partpool_Cond_H2(seed, **dat_list_onlyIDs)
#%%
print('sigma_round', prior_samples_Partpool_Cond_H2['sigma_round'].shape)
print('btreat', prior_samples_Partpool_Cond_H2['btreat'])
print('shape of delta_PSR:', prior_samples_Partpool_Cond_H2['delta_PSR_share'].shape)
#%% save them into new df
dfPrior['delta_PSR_share'] = prior_samples_Partpool_Cond_H2['delta_PSR_share'][0,:]
dfPrior['delta_SS_share'] = prior_samples_Partpool_Cond_H2['delta_SS_share'][0,:]
dfPrior['delta_BB_share'] = -(prior_samples_Partpool_Cond_H1['delta_PSR_share'][0,:] + dfPrior['delta_SS_share'])

#%% Plot sampled PSR for the different rounds
fig,ax = plt.subplots(layout='constrained')
ax.hist(dfPrior.loc[((dfPrior['r']==1)),'delta_PSR_share'].values.flatten(),bins=50, label='r1',alpha = 0.5, color='red')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==1)),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='red')
ax.hist(dfPrior.loc[((dfPrior['r']==0)),'delta_PSR_share'].values.flatten(),bins=50, label='r0',alpha = 0.3, color='blue')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==0)),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='blue')
ax.set(xlabel='Value of delta_WR', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')
#%%
fig,ax = plt.subplots(layout='constrained')
ax.hist(dfPrior.loc[(dfPrior['r']==2) ,'delta_SS_share'].values.flatten(),bins=50, label='r2',alpha = 0.5, color='green')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==2)),'delta_SS_share'].values.flatten(), alpha = 0.5, color='green')
ax.hist(dfPrior.loc[((dfPrior['r']==1)),'delta_SS_share'].values.flatten(),bins=50, label='r1',alpha = 0.5, color='red')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==1)),'delta_SS_share'].values.flatten(), alpha = 0.5, color='red')
ax.set(xlabel='Value of delta_SS', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')

#%% run MCMC with this to check
model_dummy_mcmc_H2_partpool = MCMC(NUTS(model_partpool, init_strategy = init_to_mean), num_warmup=2000, num_samples=2000, num_chains=4)
model_dummy_mcmc_H2_partpool.run(seed, r=dat_list_onlyIDs.get('r'),
    treat=dat_list_onlyIDs.get('treat'),
    delta_PSR_share = prior_samples_Partpool_Cond_H2['delta_PSR_share'][0,:], 
    delta_SS_share = prior_samples_Partpool_Cond_H2['delta_SS_share'][0,:])

mcmc_summary_dummy_H2_partpool = model_dummy_mcmc_H2_partpool.print_summary(0.89)
#%% get sample to use in plotting below
dummy_sample_H2_partpool = model_dummy_mcmc_H2_partpool.get_samples(group_by_chain=True)

#%% sigma plotting - control group over rounds
#Figure 2, right graphic
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][0,:,0,0].flatten(), label='$\sigma_{control,0}$', color='darkgreen', alpha=0.7)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][1,:,0,0].flatten(),  color='darkgreen', alpha=0.5)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][2,:,0,0].flatten(),  color='darkgreen', alpha=0.5)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][3,:,0,0].flatten(),  color='darkgreen', alpha=0.5)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][0,:,0,1].flatten(), label='$\sigma_{control,1}$', color='darkred', alpha=0.7)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][1,:,0,1].flatten(),  color='darkred', alpha=0.5)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][2,:,0,1].flatten(),  color='darkred', alpha=0.5)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][3,:,0,1].flatten(),  color='darkred', alpha=0.5)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][0,:,0,2].flatten(), label='$\sigma_{control,2}$', color='darkblue', alpha=0.7)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][1,:,0,2].flatten(),  color='darkblue', alpha=0.5)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][2,:,0,2].flatten(),  color='darkblue', alpha=0.5)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][3,:,0,2].flatten(),  color='darkblue', alpha=0.5)
ax.axvline(x=0.2, color='black', ls='--', lw=2, label='Cond.Value')
ax.axvline(x=0.15, color='black', ls='--', lw=2)
ax.axvline(x=0.1, color='black', ls='--', lw=2)
ax.set_xlabel('$\sigma_{control,r}$')
ax.set_ylabel('Density')
ax.set_xlim([0.075,.33])
ax.legend(loc='upper right')

#%% sigma plotting - treatment group over rounds
#Figure 2, right graphic
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][0,:,1,0].flatten(), label='$\sigma_{treat,0}$', color='lightgreen', alpha=0.7)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][1,:,1,0].flatten(),  color='lightgreen', alpha=0.5)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][2,:,1,0].flatten(),  color='lightgreen', alpha=0.5)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][3,:,1,0].flatten(),  color='lightgreen', alpha=0.5)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][0,:,1,1].flatten(), label='$\sigma_{treat,1}$', color='darkorange', alpha=0.7)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][1,:,1,1].flatten(),  color='darkorange', alpha=0.5)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][2,:,1,1].flatten(),  color='darkorange', alpha=0.5)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][3,:,1,1].flatten(),  color='darkorange', alpha=0.5)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][0,:,1,2].flatten(), label='$\sigma_{treat,2}$', color='lightblue', alpha=0.7)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][1,:,1,2].flatten(),  color='lightblue', alpha=0.5)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][2,:,1,2].flatten(),  color='lightblue', alpha=0.5)
sb.kdeplot(dummy_sample_H2_partpool['sigma_round'][3,:,1,2].flatten(),  color='lightblue', alpha=0.5)
ax.axvline(x=0.2, color='black', ls='--', lw=2, label='Cond.Value')
ax.axvline(x=0.17, color='black', ls='--', lw=2)
ax.axvline(x=0.12, color='black', ls='--', lw=2)
ax.set_xlabel('$\sigma_{treat,r}$')
ax.set_ylabel('Density')
ax.set_xlim([0.075,.33])
ax.legend(loc='upper right')

#%% -------------------------------- 
# Third : True H3 - variance increases over rounds, but faster for control!
# ----------------------------------
coef_trueH3_partpool = {
    'sigma_round':jnp.array([[.1, .17 ,.2],[.1 , .14, .18]]),
    'roundeff':jnp.array([[0, 0 , 0 ], [0, 0, 0]]), 
    'btreat':jnp.array([0,0],)
    }
#%% Generate synthetic data which proves this 
partpool_condModel_H3 = numpyro.handlers.condition(model_partpool, data=coef_trueH3_partpool)
prior_predictive_Partpool_Cond_H3 = Predictive(partpool_condModel_H3, num_samples=1)
#%%
prior_samples_Partpool_Cond_H3 = prior_predictive_Partpool_Cond_H3(seed, **dat_list_onlyIDs)
#%%
print('roundeff', prior_samples_Partpool_Cond_H3['roundeff'])
print('sigma_round', prior_samples_Partpool_Cond_H3['sigma_round'])
print('btreat', prior_samples_Partpool_Cond_H3['btreat'])
print('shape of delta_PSR:', prior_samples_Partpool_Cond_H3['delta_PSR_share'].shape)
#%% save them into new df
dfPrior['delta_PSR_share'] = prior_samples_Partpool_Cond_H3['delta_PSR_share'][0,:]
dfPrior['delta_SS_share'] = prior_samples_Partpool_Cond_H3['delta_SS_share'][0,:]
dfPrior['delta_BB_share'] = -(prior_samples_Partpool_Cond_H1['delta_PSR_share'][0,:] + dfPrior['delta_SS_share'])

#%% Plot sampled PSR for the different rounds
fig,ax = plt.subplots(layout='constrained')
ax.hist(dfPrior.loc[((dfPrior['r']==1)),'delta_PSR_share'].values.flatten(),bins=50, label='r1',alpha = 0.5, color='red')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==1)),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='red')
ax.hist(dfPrior.loc[((dfPrior['r']==0)),'delta_PSR_share'].values.flatten(),bins=50, label='r0',alpha = 0.3, color='blue')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==0)),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='blue')
ax.set(xlabel='Value of delta_WR', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')
#%%
fig,ax = plt.subplots(layout='constrained')
ax.hist(dfPrior.loc[(dfPrior['r']==2) ,'delta_SS_share'].values.flatten(),bins=50, label='r2',alpha = 0.5, color='green')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==2)),'delta_SS_share'].values.flatten(), alpha = 0.5, color='green')
ax.hist(dfPrior.loc[((dfPrior['r']==1)),'delta_SS_share'].values.flatten(),bins=50, label='r1',alpha = 0.5, color='red')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==1)),'delta_SS_share'].values.flatten(), alpha = 0.5, color='red')
ax.set(xlabel='Value of delta_SS', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')

#%% run MCMC with this to check
model_dummy_mcmc_H3_partpool = MCMC(NUTS(model_partpool, init_strategy = init_to_mean), num_warmup=1000, num_samples=1000, num_chains=4)
model_dummy_mcmc_H3_partpool.run(seed, r=dat_list_onlyIDs.get('r'),
    treat=dat_list_onlyIDs.get('treat'),
    delta_PSR_share = prior_samples_Partpool_Cond_H3['delta_PSR_share'][0,:], 
    delta_SS_share = prior_samples_Partpool_Cond_H3['delta_SS_share'][0,:])

mcmc_summary_dummy_H3_partpool = model_dummy_mcmc_H3_partpool.print_summary(0.89)
#%% get sample to use in plotting below
dummy_sample_H3_partpool = model_dummy_mcmc_H3_partpool.get_samples(group_by_chain=True)
#%% sigma plotting - control group over rounds
#Figure 3, left graphic
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][0,:,0,0].flatten(), label='$\sigma_{control,0}$', color='darkgreen', alpha=0.7)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][1,:,0,0].flatten(),  color='darkgreen', alpha=0.5)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][2,:,0,0].flatten(),  color='darkgreen', alpha=0.5)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][3,:,0,0].flatten(),  color='darkgreen', alpha=0.5)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][0,:,0,1].flatten(), label='$\sigma_{control,1}$', color='darkred', alpha=0.7)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][1,:,0,1].flatten(),  color='darkred', alpha=0.5)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][2,:,0,1].flatten(),  color='darkred', alpha=0.5)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][3,:,0,1].flatten(),  color='darkred', alpha=0.5)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][0,:,0,2].flatten(), label='$\sigma_{control,2}$', color='darkblue', alpha=0.7)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][1,:,0,2].flatten(),  color='darkblue', alpha=0.5)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][2,:,0,2].flatten(),  color='darkblue', alpha=0.5)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][3,:,0,2].flatten(),  color='darkblue', alpha=0.5)
ax.set_xlabel('$\sigma_{control,r}$')
ax.set_ylabel('Density')
ax.axvline(x=0.2, color='black', ls='--', lw=2, label='Cond.Value')
ax.axvline(x=0.1, color='black', ls='--', lw=2)
ax.axvline(x=0.17, color='black', ls='--', lw=2)
ax.set_xlim([0.08,0.3])
ax.legend(loc='upper right')
#%% sigma plotting - treatment group over rounds
#Figure 3, right graphic
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][0,:,1,0].flatten(), label='$\sigma_{treat,0}$', color='lightgreen', alpha=0.7)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][1,:,1,0].flatten(),  color='lightgreen', alpha=0.5)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][2,:,1,0].flatten(),  color='lightgreen', alpha=0.5)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][3,:,1,0].flatten(),  color='lightgreen', alpha=0.5)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][0,:,1,1].flatten(), label='$\sigma_{treat,1}$', color='darkorange', alpha=0.7)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][1,:,1,1].flatten(),  color='darkorange', alpha=0.5)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][2,:,1,1].flatten(),  color='darkorange', alpha=0.5)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][3,:,1,1].flatten(),  color='darkorange', alpha=0.5)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][0,:,1,2].flatten(), label='$\sigma_{treat,2}$', color='lightblue', alpha=0.7)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][1,:,1,2].flatten(),  color='lightblue', alpha=0.5)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][2,:,1,2].flatten(),  color='lightblue', alpha=0.5)
sb.kdeplot(dummy_sample_H3_partpool['sigma_round'][3,:,1,2].flatten(),  color='lightblue', alpha=0.5)
ax.set_xlabel('$\sigma_{treat,r}$')
ax.set_ylabel('Density')
ax.axvline(x=0.1, color='black', ls='--', lw=2, label='Cond.Value')
ax.axvline(x=0.14, color='black', ls='--', lw=2)
ax.axvline(x=0.18, color='black', ls='--', lw=2)
ax.set_xlim([0.08,0.3])
ax.legend(loc='upper right')

#%% -------------------------------- 
# Last: ALL H's are true AND there is a treateff
# ----------------------------------
coef_trueallH_partpool = {
    'sigma_round':jnp.array([[.15, .12 ,.2],[.15 , .14, .18]]),
    'roundeff':jnp.array([[-0.2, 0 , 0 ], [-0.2, 0, 0]]), 
    'btreat':jnp.array([0.2, 0.2],)
    }
#%% Generate synthetic data which proves this 
partpool_condModel_allH = numpyro.handlers.condition(model_partpool, data=coef_trueallH_partpool)
prior_predictive_Partpool_Cond_allH = Predictive(partpool_condModel_allH, num_samples=1)
#%%
prior_samples_Partpool_Cond_allH = prior_predictive_Partpool_Cond_allH(seed, **dat_list_onlyIDs)
#%% Plot sampled PSR for the different rounds
fig,ax = plt.subplots(layout='constrained')
ax.hist(dfPrior.loc[((dfPrior['r']==1)),'delta_PSR_share'].values.flatten(),bins=50, label='r1',alpha = 0.5, color='red')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==1)),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='red')
ax.hist(dfPrior.loc[((dfPrior['r']==0)),'delta_PSR_share'].values.flatten(),bins=50, label='r0',alpha = 0.3, color='blue')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==0)),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='blue')
ax.set(xlabel='Value of delta_WR', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')
#%%
fig,ax = plt.subplots(layout='constrained')
ax.hist(dfPrior.loc[(dfPrior['r']==2) ,'delta_SS_share'].values.flatten(),bins=50, label='r2',alpha = 0.5, color='green')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==2)),'delta_SS_share'].values.flatten(), alpha = 0.5, color='green')
ax.hist(dfPrior.loc[((dfPrior['r']==1)),'delta_SS_share'].values.flatten(),bins=50, label='r1',alpha = 0.5, color='red')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==1)),'delta_SS_share'].values.flatten(), alpha = 0.5, color='red')
ax.set(xlabel='Value of delta_SS', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')
#%%
fig,ax = plt.subplots(layout='constrained')
ax.hist(dfPrior.loc[(dfPrior['r']==0) & (dfPrior['treat']==0) ,'delta_SS_share'].values.flatten(),bins=50, label='r0, control',alpha = 0.5, color='green')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==0) & (dfPrior['treat']==0)),'delta_SS_share'].values.flatten(), alpha = 0.5, color='green')
ax.hist(dfPrior.loc[((dfPrior['r']==0) & (dfPrior['treat']==1)),'delta_SS_share'].values.flatten(),bins=50, label='r0, treat',alpha = 0.5, color='red')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==0) & (dfPrior['treat']==1)),'delta_SS_share'].values.flatten(), alpha = 0.5, color='red')
ax.set(xlabel='Value of delta_SS', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')

#%% run MCMC with this to check
model_dummy_mcmc_allH_partpool = MCMC(NUTS(model_partpool, init_strategy = init_to_median), num_warmup=1000, num_samples=1000, num_chains=4)
model_dummy_mcmc_allH_partpool.run(seed, r=dat_list_onlyIDs.get('r'),
    treat=dat_list_onlyIDs.get('treat'),
    delta_PSR_share = prior_samples_Partpool_Cond_allH['delta_PSR_share'][0,:], 
    delta_SS_share = prior_samples_Partpool_Cond_allH['delta_SS_share'][0,:])

mcmc_summary_dummy_allH_partpool = model_dummy_mcmc_allH_partpool.print_summary(0.89)

#%% get sample to use in plotting below
dummy_sample_allH_partpool = model_dummy_mcmc_allH_partpool.get_samples(group_by_chain=True)
#%% Plot Posteriors for the roundeffect coefficient
# Figure 4, First row
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][1,:,0,0].flatten(), label='$\\theta_{WR,0}$', color='green', alpha=0.7)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][2,:,0,0].flatten(),  color='green', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][3,:,0,0].flatten(),  color='green', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][0,:,0,0].flatten(),  color='green', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][0,:,1,0].flatten(), label='$\\theta_{SS,1}$', color='purple', alpha=0.7)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][1,:,1,0].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][2,:,1,0].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][3,:,1,0].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][1,:,0,1].flatten(), label='$\\theta_{WR,1}$', color='red', alpha=0.7)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][2,:,0,1].flatten(),  color='red', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][3,:,0,1].flatten(),  color='red', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][0,:,0,1].flatten(),  color='red', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][0,:,1,1].flatten(), label='$\\theta_{SS,1}$', color='orange', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][1,:,1,1].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][2,:,1,1].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][3,:,1,1].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][1,:,0,2].flatten(), label='$\\theta_{WR,2}$', color='blue', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][2,:,0,2].flatten(),  color='blue', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][3,:,0,2].flatten(),  color='blue', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][0,:,0,2].flatten(),  color='blue', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][0,:,1,2].flatten(), label='$\\theta_{SS,2}$', color='grey', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][1,:,1,2].flatten(),  color='grey', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][2,:,1,2].flatten(),  color='grey', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][3,:,1,2].flatten(),  color='grey', alpha=0.5)
ax.axvline(x=-0.2, color='black', ls='--', lw=2, label='Cond. Value')
ax.axvline(x=0, color='black', ls='--', lw=2)
#ax.set_title('Posterior Density for the $roundeffect$ Parameter')    
ax.set_xlabel('$\\theta_{r,tech}$')
ax.set_ylabel('Density')
ax.set_xlim([-0.27,0.25])
ax.legend(loc='upper right')

#%% Plot Posteriors for the sigma coefficient of the control group
# Figure 4, second row, left graphic
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][0,:,0,0].flatten(), label='$\sigma_{control,0}$', color='darkgreen', alpha=0.7)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][1,:,0,0].flatten(),  color='darkgreen', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][2,:,0,0].flatten(),  color='darkgreen', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][3,:,0,0].flatten(),  color='darkgreen', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][0,:,0,1].flatten(), label='$\sigma_{control,1}$', color='darkred', alpha=0.7)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][1,:,0,1].flatten(),  color='darkred', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][2,:,0,1].flatten(),  color='darkred', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][3,:,0,1].flatten(),  color='darkred', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][0,:,0,2].flatten(), label='$\sigma_{control,2}$', color='darkblue', alpha=0.7)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][1,:,0,2].flatten(),  color='darkblue', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][2,:,0,2].flatten(),  color='darkblue', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][3,:,0,2].flatten(),  color='darkblue', alpha=0.5)
ax.axvline(x=0.15, color='black', ls='--', lw=2, label='Cond. Value')
ax.axvline(x=0.12, color='black', ls='--', lw=2)
ax.axvline(x=0.2, color='black', ls='--', lw=2)
ax.set_xlabel('$\sigma_{control,r}$')
ax.set_ylabel('Density')
ax.set_xlim([0.08,.3])
ax.legend(loc='upper right')

#%% Plot Posteriors for the sigma coefficient of the treatment group
# Figure 4, second row, right graphic
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][0,:,1,0].flatten(), label='$\sigma_{treat,0}$', color='lightgreen')
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][1,:,1,0].flatten(),  color='lightgreen', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][2,:,1,0].flatten(),  color='lightgreen', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][3,:,1,0].flatten(),  color='lightgreen', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][0,:,1,1].flatten(), label='$\sigma_{treat,1}$', color='darkorange')
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][1,:,1,1].flatten(),  color='darkorange', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][2,:,1,1].flatten(),  color='darkorange', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][3,:,1,1].flatten(),  color='darkorange', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][0,:,1,2].flatten(), label='$\sigma_{treat,2}$', color='lightblue')
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][1,:,1,2].flatten(),  color='lightblue', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][2,:,1,2].flatten(),  color='lightblue', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['sigma_round'][3,:,1,2].flatten(),  color='lightblue', alpha=0.5)
ax.axvline(x=0.15, color='black', ls='--', lw=2, label='Cond. Value')
ax.axvline(x=0.14, color='black', ls='--', lw=2)
ax.axvline(x=0.18, color='black', ls='--', lw=2)
ax.set_xlabel('$\sigma_{treat,r}$')
ax.set_ylabel('Density')
ax.set_xlim([0.08,.3])
ax.legend(loc='upper right')

#%% treateff posterior for no model WITH roundeffs
#Figure 5, left graphic
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(dummy_sample_allH_partpool['btreat'][0,:,0].flatten(), label='$\\beta_{WR}$', color='orange', alpha=0.7)
sb.kdeplot(dummy_sample_allH_partpool['btreat'][1,:,0].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['btreat'][2,:,0].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['btreat'][3,:,0].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['btreat'][0,:,1].flatten(), label='$\\beta_{SS}$', color='purple', alpha=0.7)
sb.kdeplot(dummy_sample_allH_partpool['btreat'][1,:,1].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['btreat'][2,:,1].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['btreat'][3,:,1].flatten(),  color='purple', alpha=0.5)
ax.set_xlabel('$\\beta_{tech}$')  
ax.axvline(x=0.2, color='black', ls='--', lw=2, label='Cond.Value')
ax.set_ylabel('Density')
ax.set_xlim([0.1,0.3])
ax.legend(loc='upper right')

#%% -----------------------------------------------------------------------------
# compare this model estimation with model without roundeffect estimation
#--------------------------------------------------------------------------------
#%%
def model_noRoundeff(treat, delta_PSR_share=None, delta_SS_share=None): 
    
    btreat = numpyro.sample("btreat", dist.Normal(0, 0.3).expand([2]))

    mu_delta_PSR = btreat[0]*treat
    mu_delta_SS = btreat[1]*treat

    sigma = numpyro.sample('sigma', dist.Exponential(1))

    delta_PSR_share = numpyro.sample('delta_PSR_share', dist.TruncatedNormal(loc=mu_delta_PSR, scale=sigma,low=-1, high=1), obs=delta_PSR_share)
    delta_SS_share = numpyro.sample('delta_SS_share', dist.TruncatedNormal(loc=mu_delta_SS, scale=sigma,low=-1, high=1), obs=delta_SS_share)

#%% feeding this model with data generated based on conditioning that all hypotheses are true and treateffect is :
# 'btreat':jnp.array([.2,.2],)
    
model_dummy_mcmc_noRound = MCMC(NUTS(model_noRoundeff, init_strategy = init_to_median), num_warmup=1000, num_samples=1000, num_chains=4)
model_dummy_mcmc_noRound.run(random.PRNGKey(0), 
    treat=dat_list_onlyIDs.get('treat'),
    delta_PSR_share = prior_samples_Partpool_Cond_allH['delta_PSR_share'][0,:], 
    delta_SS_share = prior_samples_Partpool_Cond_allH['delta_SS_share'][0,:])

mcmc_summary = model_dummy_mcmc_noRound.print_summary(0.89)

#%% compare posteriors of treateff for both models
samples_mcmc_noRound = model_dummy_mcmc_noRound.get_samples(group_by_chain=True)
#%% treateff posterior for no roundeff model
#Figure 5, right graphic
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(samples_mcmc_noRound['btreat'][0,:,0].flatten(), label='$\\beta_{WR}$', color='orange', alpha=0.7)
sb.kdeplot(samples_mcmc_noRound['btreat'][1,:,0].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(samples_mcmc_noRound['btreat'][2,:,0].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(samples_mcmc_noRound['btreat'][3,:,0].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(samples_mcmc_noRound['btreat'][0,:,1].flatten(), label='$\\beta_{SS}$', color='purple', alpha=0.7)
sb.kdeplot(samples_mcmc_noRound['btreat'][1,:,1].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(samples_mcmc_noRound['btreat'][2,:,1].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(samples_mcmc_noRound['btreat'][3,:,1].flatten(),  color='purple', alpha=0.5)
ax.set_xlabel('$\\beta_{tech}$')  
ax.axvline(x=0.2, color='black', ls='--', lw=2, label='Cond.Value')
ax.set_ylabel('Density')
ax.set_xlim([0,0.3])
ax.legend(loc='upper right')

#%% treateff posterior for no roundeff model
#erste dimension: chain
#zweite dimension: sample draw, immer alle includen
#dritte dimension: runde
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(samples_mcmc_noRound['sigma'][0,:].flatten(), label='$\sigma$', color='grey', alpha=0.7)
sb.kdeplot(samples_mcmc_noRound['sigma'][1,:].flatten(),  color='grey', alpha=0.5)
sb.kdeplot(samples_mcmc_noRound['sigma'][2,:].flatten(),  color='grey', alpha=0.5)
sb.kdeplot(samples_mcmc_noRound['sigma'][3,:].flatten(),  color='grey', alpha=0.5)
ax.set_xlabel('$\sigma$')  
ax.set_ylabel('Density')
#ax.set_xlim([0,0.3])
ax.legend(loc='upper right')


#%% -----------------------------------------------------------------------------
#--------------RUN MODEL WITH REAL DATA -----------------------------------------
#--------------------------------------------------------------------------------
#%% Create a dictionary with 'real' Data including output variables
# - note that we only keep two out of the three included technologies because the third can be expressed as the 'rest'
dat_list_allOutputs_shares = dict(
    r=jnp.array([*jnp.tile(0, N),*jnp.tile(1, N),*jnp.tile(2, N)]),
    treat=jnp.tile(jnp.array([*jnp.tile(0, int(N/2)),*jnp.tile(1, int(N/2))]), n_r),
    delta_PSR_share= dist.TruncatedNormal(loc=0, low= -1, high=1,scale=0.3).sample(seed, (1,N*n_r)),
    delta_SS_share= dist.TruncatedNormal(loc=0, low= -1, high=1, scale=0.3).sample(seed, (1,N*n_r)),
)
dat_list_allOutputs_shares.get('delta_PSR_share').shape

#%% Run the MCMC with this real data
model_partpool_realDat = MCMC(NUTS(model_partpool, init_strategy = init_to_median), num_warmup=1000, 
num_samples=1000, num_chains=4)
model_partpool_realDat.run(random.PRNGKey(0), **dat_list_allOutputs_shares)

mcmc_summary_partpool = model_partpool_realDat.print_summary(0.89)
#%%  
samples_partpool_realDat = model_partpool_realDat.get_samples(group_by_chain=True)
#%% traceplot for chain inspection and parameter estimate overview
az.plot_trace(samples_partpool_realDat)

#%% plot roundeff parameter
# Figure 6 replication with generated data
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(samples_partpool_realDat['roundeff'][1,:,0,0].flatten(), label='$\\theta_{WR,0}$', color='green', alpha=0.7)
sb.kdeplot(samples_partpool_realDat['roundeff'][2,:,0,0].flatten(),  color='green', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][3,:,0,0].flatten(),  color='green', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][0,:,0,0].flatten(),  color='green', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][0,:,1,0].flatten(), label='$\\theta_{SS,0}$', color='purple', alpha=0.7)
sb.kdeplot(samples_partpool_realDat['roundeff'][1,:,1,0].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][2,:,1,0].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][3,:,1,0].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][1,:,0,1].flatten(), label='$\\theta_{WR,1}$', color='red', alpha=0.7)
sb.kdeplot(samples_partpool_realDat['roundeff'][2,:,0,1].flatten(),  color='red', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][3,:,0,1].flatten(),  color='red', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][0,:,0,1].flatten(),  color='red', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][0,:,1,1].flatten(), label='$\\theta_{SS,1}$', color='orange', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][1,:,1,1].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][2,:,1,1].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][3,:,1,1].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][1,:,0,2].flatten(), label='$\\theta_{WR,2}$', color='blue', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][2,:,0,2].flatten(),  color='blue', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][3,:,0,2].flatten(),  color='blue', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][0,:,0,2].flatten(),  color='blue', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][0,:,1,2].flatten(), label='$\\theta_{SS,2}$', color='grey', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][1,:,1,2].flatten(),  color='grey', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][2,:,1,2].flatten(),  color='grey', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['roundeff'][3,:,1,2].flatten(),  color='grey', alpha=0.5)
#ax.set_title('Posterior Density for the $roundeffect$ Parameter')    
ax.set_xlabel('$\\theta_{r,tech}$')
ax.set_ylabel('Density')
ax.set_xlim([-0.15,0.2])
ax.legend(loc='upper right')
#%% Sigma
# Figure 7, left graphic replication with generated data
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(samples_partpool_realDat['sigma_round'][0,:,0,0].flatten(), label='$\sigma_{control,0}$', color='darkgreen', alpha=0.7)
sb.kdeplot(samples_partpool_realDat['sigma_round'][1,:,0,0].flatten(),  color='darkgreen', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['sigma_round'][2,:,0,0].flatten(),  color='darkgreen', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['sigma_round'][3,:,0,0].flatten(),  color='darkgreen', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['sigma_round'][0,:,0,1].flatten(), label='$\sigma_{control,1}$', color='darkred', alpha=0.7)
sb.kdeplot(samples_partpool_realDat['sigma_round'][1,:,0,1].flatten(),  color='darkred', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['sigma_round'][2,:,0,1].flatten(),  color='darkred', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['sigma_round'][3,:,0,1].flatten(),  color='darkred', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['sigma_round'][0,:,0,2].flatten(), label='$\sigma_{control,2}$', color='darkblue', alpha=0.7)
sb.kdeplot(samples_partpool_realDat['sigma_round'][1,:,0,2].flatten(),  color='darkblue', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['sigma_round'][2,:,0,2].flatten(),  color='darkblue', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['sigma_round'][3,:,0,2].flatten(),  color='darkblue', alpha=0.5)
ax.set_xlabel('$\sigma_{control,r}$')
ax.set_ylabel('Density')
ax.legend(loc='upper right')
#%% Sigma
# Figure 7, right graphic replication with generated data
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(samples_partpool_realDat['sigma_round'][0,:,1,0].flatten(), label='$\sigma_{treat,0}$', color='lightgreen')
sb.kdeplot(samples_partpool_realDat['sigma_round'][1,:,1,0].flatten(),  color='lightgreen', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['sigma_round'][2,:,1,0].flatten(),  color='lightgreen', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['sigma_round'][3,:,1,0].flatten(),  color='lightgreen', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['sigma_round'][0,:,1,1].flatten(), label='$\sigma_{treat,1}$', color='darkorange')
sb.kdeplot(samples_partpool_realDat['sigma_round'][1,:,1,1].flatten(),  color='darkorange', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['sigma_round'][2,:,1,1].flatten(),  color='darkorange', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['sigma_round'][3,:,1,1].flatten(),  color='darkorange', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['sigma_round'][0,:,1,2].flatten(), label='$\sigma_{treat,2}$', color='lightblue')
sb.kdeplot(samples_partpool_realDat['sigma_round'][1,:,1,2].flatten(),  color='lightblue', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['sigma_round'][2,:,1,2].flatten(),  color='lightblue', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['sigma_round'][3,:,1,2].flatten(),  color='lightblue', alpha=0.5)
ax.set_xlabel('$\sigma_{treat,r}$')
ax.set_ylabel('Density')
ax.legend(loc='upper right')

#%% treateff posterior 
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(samples_partpool_realDat['btreat'][0,:,0].flatten(), label='$\\beta_{WR}$', color='orange', alpha=0.7)
sb.kdeplot(samples_partpool_realDat['btreat'][1,:,0].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['btreat'][2,:,0].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['btreat'][3,:,0].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['btreat'][0,:,1].flatten(), label='$\\beta_{SS}$', color='purple', alpha=0.7)
sb.kdeplot(samples_partpool_realDat['btreat'][1,:,1].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['btreat'][2,:,1].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['btreat'][3,:,1].flatten(),  color='purple', alpha=0.5)
ax.set_xlabel('$\\beta_{tech}$')  
ax.set_ylabel('Density')
ax.legend(loc='upper right')

