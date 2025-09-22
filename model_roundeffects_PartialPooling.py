#%%
import os
import warnings

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

import numpy as np

import jax.numpy as jnp
from jax import lax, random
from jax.scipy.special import expit

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import effective_sample_size
from numpyro.infer import MCMC, NUTS, Predictive, init_to_mean, init_to_median

if "png" in os.environ:
    %config InlineBackend.figure_formats = ["png"]
warnings.formatwarning = lambda message, category, *args, **kwargs: "{}: {}\n".format(
    category.__name__, message
)
az.style.use("arviz-darkgrid")
numpyro.set_platform("cpu")

#%% ------------------------------------------------ 
# Load the Dataframe and get some first impressions 
# -------------------------------------------------- 
url = 'https://phenoroam.phenorob.de/geonetwork/srv/api/records/979019ef-037a-4716-9c40-dfa0125f16f4/attachments/Dataset_RoundEffects.csv'
Data = pd.read_csv(url)
#%% Visualize the responses of the control group across rounds 
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(Data.loc[((Data['r']==2)& Data['treat']==0),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='green')
sb.kdeplot(Data.loc[((Data['r']==1)& Data['treat']==0),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='red')
sb.kdeplot(Data.loc[((Data['r']==0)& Data['treat']==0),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='blue')
ax.set(xlabel='Value of delta_WR', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right', handles=[])
#%% Visualize the responses of the Treatment group across rounds 
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(Data.loc[((Data['r']==2)& Data['treat']==1),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='green')
sb.kdeplot(Data.loc[((Data['r']==1)& Data['treat']==1),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='red')
sb.kdeplot(Data.loc[((Data['r']==0)& Data['treat']==1),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='blue')
ax.set(xlabel='Value of delta_WR', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')
#%% -------------------------------------------------
# Define Dictionaries for Conditioning and Inference
# -------------------------------------------------
#%% Create dictionary with only the Data Structure, excluding actual Responses

dat_list_onlyIDs = dict(
    r=Data.r.values,
    treat=Data.treat.values
)
dat_list_onlyIDs
#%% Create dictionary from the actual Dataframe, including actual Responses

dat_list_allOutputs = dict(
    r=Data.r.values,
    treat=Data.treat.values,
    delta_WR = Data.delta_PSR_share.values,
    delta_SS = Data.delta_SS_share.values,
)
Data.r.values

#%% ------------------------------------------------------------------------------
# Partially pooled model with varying sigma and theta per treatment group
# --------------------------------------------------------------------------------
#includes meta-level where sigma priors depend on the treatment group

def model_partpool(r, treat, delta_WR=None, delta_SS=None): 
    
    # Set a prior for the switching parameter 'theta' 
    theta = numpyro.sample("theta", dist.Normal(0, 0.3), sample_shape=(2,3)) #per technology and round

    # Set a partially pooled prior for the variance 'sigma' of response changes per round
    sigma_bar = numpyro.sample("sigma_bar", dist.TruncatedNormal(loc=1, scale=0.2, low=0))
    sigma_round = numpyro.sample("sigma_round", dist.Exponential(sigma_bar), sample_shape=(2,3)) #per round and treatment group

    # Set a prior for the treatment effect 'beta'
    beta_treat = numpyro.sample("beta_treat", dist.Normal(0, 0.3).expand([2])) #per technology

    # Define the average response values to be used for drawing the posterior distributions
    mu_delta_PSR = theta[0,r] + beta_treat[0]*treat
    mu_delta_SS = theta[1,r] + beta_treat[1]*treat

    # To enable round- and treatment-wise sigma-definition, save this as an own arraw to be used for drawing the posteriors
    sigma = sigma_round[treat, r]

    # Used the averages and the variance defenitions from above to draw posterior distributions for the response variables
    delta_WR = numpyro.sample("delta_WR", dist.TruncatedNormal(loc=mu_delta_PSR, scale=sigma,low=-1, high=1), obs=delta_WR)
    delta_SS = numpyro.sample("delta_SS", dist.TruncatedNormal(loc=mu_delta_SS, scale=sigma,low=-1, high=1), obs=delta_SS)

#%% ------------------------------------------------------------------------------
# Use conditioned Data generation to test and inspect the models functionality
# --------------------------------------------------------------------------------
#%% ------------------------------------------------------------------------------
# First Conditioning
# True H1: Theta in r=0 is non-zero
# --------------------------------------------------------------------------------
coef_trueH1 = {
    'sigma_round':jnp.array([[.2, .2 ,.2],[.2 , .2, .2 ]]),
    'theta':jnp.array([[-0.2,0,0],[-0.2,0,0]]),
    'beta_treat':jnp.array([0.,0.],)
    }
#%% Generate synthetic data which aligns with these coefficient values
# Note: Use the dictionary without response data here, just pass on information about the data structure
condModel_H1 = numpyro.handlers.condition(model_partpool, data=coef_trueH1)
prior_predictive_Cond_H1 = Predictive(condModel_H1, num_samples=1)
prior_samples_Cond_H1 = prior_predictive_Cond_H1(random.PRNGKey(0), **dat_list_onlyIDs) 
#%% inspect whether the conditioned coefficient values where indeed maintained
print('theta', prior_samples_Cond_H1['theta'])
print('sigma_round', prior_samples_Cond_H1['sigma_round'][0,0])
print('beta_treat', prior_samples_Cond_H1['beta_treat'])
#%% Use the generated data to run the 'inference procedure' (passing it into the MCMC)
model_dummy_mcmc_H1 = MCMC(NUTS(model_partpool, init_strategy = init_to_median), num_warmup=1000, num_samples=1000, num_chains=4)
model_dummy_mcmc_H1.run(random.PRNGKey(0), 
    r=Data.r.values,
    treat=Data.treat.values,
    delta_WR = prior_samples_Cond_H1['delta_WR'][0,:], 
    delta_SS = prior_samples_Cond_H1['delta_SS'][0,:])
#%% Inspect the coefficient information to check whether it aligns with conditioned parameter values
mcmc_summary_dummy_H1 = model_dummy_mcmc_H1.print_summary(0.89)
#%% Draw from the posterior to generate a sample to be used for visualization
dummy_sample_H1_posteriorSample = model_dummy_mcmc_H1.get_samples(group_by_chain=True)
#%% Save the conditioned data into a dataframe to ease visualization
dfPrior_condH1 = Data.copy()
dfPrior_condH1['delta_WR'] = prior_samples_Cond_H1['delta_WR'][0,:]
dfPrior_condH1['delta_SS'] = prior_samples_Cond_H1['delta_SS'][0,:]
dfPrior_condH1['delta_BB'] = -(dfPrior_condH1['delta_WR'] + dfPrior_condH1['delta_SS'])
#%% Visualize the generated data
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
# Figure 2, first row, left plot
ax[0,0].hist(dfPrior_condH1.loc[((dfPrior_condH1['r']==1)),'delta_WR'].values.flatten(),bins=50, label='r1',alpha = 0.5, color='red')
sb.kdeplot(dfPrior_condH1.loc[((dfPrior_condH1['r']==1)),'delta_WR'].values.flatten(), alpha = 0.5, color='red', ax=ax[0,0])
ax[0,0].hist(dfPrior_condH1.loc[((dfPrior_condH1['r']==0)),'delta_WR'].values.flatten(),bins=50, label='r0',alpha = 0.3, color='blue')
sb.kdeplot(dfPrior_condH1.loc[((dfPrior_condH1['r']==0)),'delta_WR'].values.flatten(), alpha = 0.5, color='blue', ax=ax[0,0])
ax[0,0].set(xlabel='Value of delta_WR', ylabel='Frequency in the dummy data')
ax[0,0].legend(loc='upper right')
# Figure 2, first row, right plot
ax[0,1].hist(dfPrior_condH1.loc[(dfPrior_condH1['r']==2) ,'delta_SS'].values.flatten(),bins=50, label='r2',alpha = 0.5, color='green')
sb.kdeplot(dfPrior_condH1.loc[((dfPrior_condH1['r']==2)),'delta_SS'].values.flatten(), alpha = 0.5, color='green', ax=ax[0,1])
ax[0,1].hist(dfPrior_condH1.loc[((dfPrior_condH1['r']==1)),'delta_SS'].values.flatten(),bins=50, label='r1',alpha = 0.5, color='red')
sb.kdeplot(dfPrior_condH1.loc[((dfPrior_condH1['r']==1)),'delta_SS'].values.flatten(), alpha = 0.5, color='red', ax=ax[0,1])
ax[0,1].set(xlabel='Value of delta_SS', ylabel='Frequency in the dummy data')
ax[0,1].legend(loc='upper right')
# Visualize the obtained parameter values in comparison to the pre-defined conditioned values
# Shape of the posterior sample coefficient arrays: [chain, sample draw, technology, round]
# Figure 2, second row, left plot
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][0,:,0,0].flatten(), label='$\sigma_{control,0}$', color='darkgreen', alpha=0.7, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][1,:,0,0].flatten(),  color='darkgreen', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][2,:,0,0].flatten(),  color='darkgreen', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][3,:,0,0].flatten(),  color='darkgreen', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][0,:,0,1].flatten(), label='$\sigma_{control,1}$', color='darkred', alpha=0.7, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][1,:,0,1].flatten(),  color='darkred', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][2,:,0,1].flatten(),  color='darkred', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][3,:,0,1].flatten(),  color='darkred', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][0,:,0,2].flatten(), label='$\sigma_{control,2}$', color='darkblue', alpha=0.7, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][1,:,0,2].flatten(),  color='darkblue', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][2,:,0,2].flatten(),  color='darkblue', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][3,:,0,2].flatten(),  color='darkblue', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][0,:,1,0].flatten(), label='$\sigma_{treat,0}$', color='lightgreen', alpha=0.7, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][1,:,1,0].flatten(),  color='lightgreen', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][2,:,1,0].flatten(),  color='lightgreen', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][3,:,1,0].flatten(),  color='lightgreen', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][0,:,1,1].flatten(), label='$\sigma_{treat,1}$', color='darkorange', alpha=0.7, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][1,:,1,1].flatten(),  color='darkorange', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][2,:,1,1].flatten(),  color='darkorange', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][3,:,1,1].flatten(),  color='darkorange', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][0,:,1,2].flatten(), label='$\sigma_{treat,2}$', color='lightblue', alpha=0.7, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][1,:,1,2].flatten(),  color='lightblue', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][2,:,1,2].flatten(),  color='lightblue', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_H1_posteriorSample['sigma_round'][3,:,1,2].flatten(),  color='lightblue', alpha=0.5, ax=ax[1,0])
ax[1,0].axvline(x=0.2, color='black', ls='--', lw=2, label='Cond.Value')
ax[1,0].set_xlabel('$\sigma_{treat,r}$')
ax[1,0].set_ylabel('Density')
ax[1,0].set_xlim([0.12,0.35])
ax[1,0].legend(loc='upper right')
# Figure 2, second row, right plot
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][1,:,0,0].flatten(), label='$\\theta_{WR,0}$', color='green', alpha=0.7, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][2,:,0,0].flatten(),  color='green', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][3,:,0,0].flatten(),  color='green', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][0,:,0,0].flatten(),  color='green', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][1,:,0,1].flatten(), label='$\\theta_{WR,1}$', color='red', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][2,:,0,1].flatten(),  color='red', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][3,:,0,1].flatten(),  color='red', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][0,:,0,1].flatten(),  color='red', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][1,:,0,2].flatten(), label='$\\theta_{WR,2}$', color='blue', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][2,:,0,2].flatten(),  color='blue', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][3,:,0,2].flatten(),  color='blue', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][0,:,0,2].flatten(),  color='blue', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][0,:,1,0].flatten(), label='$\\theta_{SS,0}$', color='purple', alpha=0.7, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][1,:,1,0].flatten(),  color='purple', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][2,:,1,0].flatten(),  color='purple', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][3,:,1,0].flatten(),  color='purple', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][0,:,1,1].flatten(), label='$\\theta_{SS,1}$', color='orange', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][1,:,1,1].flatten(),  color='orange', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][2,:,1,1].flatten(),  color='orange', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][3,:,1,1].flatten(),  color='orange', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][0,:,1,2].flatten(), label='$\\theta_{SS,2}$', color='grey', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][1,:,1,2].flatten(),  color='grey', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][2,:,1,2].flatten(),  color='grey', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_H1_posteriorSample['theta'][3,:,1,2].flatten(),  color='grey', alpha=0.5, ax=ax[1,1])
ax[1,1].axvline(x=-0.2, color='black', ls='--', lw=2, label='Cond.Value')
ax[1,1].axvline(x=0, color='black', ls='--', lw=2)  
ax[1,1].set_xlabel('$\\theta_{r,tech}$')
ax[1,1].set_ylabel('Density')
ax[1,1].set_xlim([-0.3,0.25])
ax[1,1].legend(loc='upper right')
#%% Optional: Export als png
#fig.savefig("Figure2.png", format="png")
#%% ------------------------------------------------------------------------------
# Second Conditioning
# True H2: Sigma decreases over rounds and faster for the control group
# --------------------------------------------------------------------------------
coef_trueH2 = {
    'sigma_round':jnp.array([[.2, .15 ,.1],[.2 , .17, .12 ]]),
    'theta':jnp.array([[0,0,0],[0,0,0]]),
    'beta_treat':jnp.array([0.,0.],)
    }
#%% Generate synthetic data which aligns with these coefficient values
# Note: Use the dictionary without response data here, just pass on information about the data structure
condModel_H2 = numpyro.handlers.condition(model_partpool, data=coef_trueH2)
prior_predictive_Cond_H2 = Predictive(condModel_H2, num_samples=1)
prior_samples_Cond_H2 = prior_predictive_Cond_H2(random.PRNGKey(0), **dat_list_onlyIDs) 
#%% inspect whether the conditioned coefficient values where indeed maintained
print('theta', prior_samples_Cond_H2['theta'])
print('sigma_round', prior_samples_Cond_H2['sigma_round'][0,0])
print('beta_treat', prior_samples_Cond_H2['beta_treat'])
#%% Save the conditioned data into a dataframe to ease visualization
dfPrior_condH2 = Data.copy()
dfPrior_condH2['delta_WR'] = prior_samples_Cond_H2['delta_WR'][0,:]
dfPrior_condH2['delta_SS'] = prior_samples_Cond_H2['delta_SS'][0,:]
dfPrior_condH2['delta_BB'] = -(dfPrior_condH2['delta_WR'] + dfPrior_condH2['delta_SS'])
#%% Visualize the generated data
fig,ax = plt.subplots(layout='constrained')
ax.hist(dfPrior_condH2.loc[((dfPrior_condH2['r']==2) & dfPrior_condH2['treat']==1),'delta_PSR_share'].values.flatten(),bins=50, label='r2',alpha = 0.5, color='green')
sb.kdeplot(dfPrior_condH2.loc[((dfPrior_condH2['r']==2)& dfPrior_condH2['treat']==1),'delta_PSR_share'].values.flatten(),alpha = 0.5, color='green')
ax.hist(dfPrior_condH2.loc[((dfPrior_condH2['r']==1)& dfPrior_condH2['treat']==1),'delta_PSR_share'].values.flatten(),bins=50, label='r1',alpha = 0.5, color='red')
sb.kdeplot(dfPrior_condH2.loc[((dfPrior_condH2['r']==1)& dfPrior_condH2['treat']==1),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='red')
ax.hist(dfPrior_condH2.loc[((dfPrior_condH2['r']==0)& dfPrior_condH2['treat']==1),'delta_PSR_share'].values.flatten(),bins=50, label='r0',alpha = 0.3, color='blue')
sb.kdeplot(dfPrior_condH2.loc[((dfPrior_condH2['r']==0)& dfPrior_condH2['treat']==1),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='blue')
ax.set(xlabel='Value of delta_WR', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')
#%% 
fig,ax = plt.subplots(layout='constrained')
ax.hist(dfPrior_condH2.loc[((dfPrior_condH2['r']==2) & dfPrior_condH2['treat']==0),'delta_PSR_share'].values.flatten(),bins=50, label='r2',alpha = 0.5, color='green')
sb.kdeplot(dfPrior_condH2.loc[((dfPrior_condH2['r']==2)& dfPrior_condH2['treat']==0),'delta_PSR_share'].values.flatten(),alpha = 0.5, color='green')
ax.hist(dfPrior_condH2.loc[((dfPrior_condH2['r']==1)& dfPrior_condH2['treat']==0),'delta_PSR_share'].values.flatten(),bins=50, label='r1',alpha = 0.5, color='red')
sb.kdeplot(dfPrior_condH2.loc[((dfPrior_condH2['r']==1)& dfPrior_condH2['treat']==0),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='red')
ax.hist(dfPrior_condH2.loc[((dfPrior_condH2['r']==0)& dfPrior_condH2['treat']==0),'delta_PSR_share'].values.flatten(),bins=50, label='r0',alpha = 0.3, color='blue')
sb.kdeplot(dfPrior_condH2.loc[((dfPrior_condH2['r']==0)& dfPrior_condH2['treat']==0),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='blue')
ax.set(xlabel='Value of delta_WR', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')
#%% Use the generated data to run the 'inference procedure' (passing it into the MCMC)
model_dummy_mcmc_H2 = MCMC(NUTS(model_partpool, init_strategy = init_to_median), num_warmup=1000, num_samples=1000, num_chains=4)
model_dummy_mcmc_H2.run(random.PRNGKey(0), 
    r=Data.r.values,
    treat=Data.treat.values,
    delta_WR = prior_samples_Cond_H2['delta_WR'][0,:], 
    delta_SS = prior_samples_Cond_H2['delta_SS'][0,:])
#%% Inspect the coefficient information to check whether it aligns with conditioned parameter values
mcmc_summary_dummy_H2 = model_dummy_mcmc_H2.print_summary(0.89)
#%% Draw from the posterior to generate a sample to be used for visualization
dummy_sample_H2_posteriorSample = model_dummy_mcmc_H2.get_samples(group_by_chain=True)
#%% Visualize the obtained parameter values in comparison to the pre-defined conditioned values
fig, ax = plt.subplots(1, 2, figsize=(15, 5), layout='constrained')
# Figure 3, left plot
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][0,:,0,0].flatten(), label='$\sigma_{control,0}$', color='darkgreen', alpha=0.7, ax=ax[0])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][1,:,0,0].flatten(),  color='darkgreen', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][2,:,0,0].flatten(),  color='darkgreen', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][3,:,0,0].flatten(),  color='darkgreen', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][0,:,0,1].flatten(), label='$\sigma_{control,1}$', color='darkred', alpha=0.7, ax=ax[0])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][1,:,0,1].flatten(),  color='darkred', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][2,:,0,1].flatten(),  color='darkred', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][3,:,0,1].flatten(),  color='darkred', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][0,:,0,2].flatten(), label='$\sigma_{control,2}$', color='darkblue', alpha=0.7, ax=ax[0])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][1,:,0,2].flatten(),  color='darkblue', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][2,:,0,2].flatten(),  color='darkblue', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][3,:,0,2].flatten(),  color='darkblue', alpha=0.5, ax=ax[0])
ax[0].axvline(x=0.2, color='black', ls='--', lw=2, label='Cond.Value')
ax[0].axvline(x=0.15, color='black', ls='--', lw=2)
ax[0].axvline(x=0.1, color='black', ls='--', lw=2)
ax[0].set_xlabel('$\sigma_{control,r}$')
ax[0].set_ylabel('Density')
ax[0].set_xlim([0.075,.33])
ax[0].legend(loc='upper right')
# Figure 3, right plot
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][0,:,1,0].flatten(), label='$\sigma_{treat,0}$', color='lightgreen', alpha=0.7, ax=ax[1])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][1,:,1,0].flatten(),  color='lightgreen', alpha=0.5, ax=ax[1])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][2,:,1,0].flatten(),  color='lightgreen', alpha=0.5, ax=ax[1])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][3,:,1,0].flatten(),  color='lightgreen', alpha=0.5, ax=ax[1])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][0,:,1,1].flatten(), label='$\sigma_{treat,1}$', color='darkorange', alpha=0.7, ax=ax[1])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][1,:,1,1].flatten(),  color='darkorange', alpha=0.5, ax=ax[1])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][2,:,1,1].flatten(),  color='darkorange', alpha=0.5, ax=ax[1])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][3,:,1,1].flatten(),  color='darkorange', alpha=0.5, ax=ax[1])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][0,:,1,2].flatten(), label='$\sigma_{treat,2}$', color='lightblue', alpha=0.7, ax=ax[1])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][1,:,1,2].flatten(),  color='lightblue', alpha=0.5, ax=ax[1])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][2,:,1,2].flatten(),  color='lightblue', alpha=0.5, ax=ax[1])
sb.kdeplot(dummy_sample_H2_posteriorSample['sigma_round'][3,:,1,2].flatten(),  color='lightblue', alpha=0.5, ax=ax[1])
ax[1].axvline(x=0.2, color='black', ls='--', lw=2, label='Cond.Value')
ax[1].axvline(x=0.17, color='black', ls='--', lw=2)
ax[1].axvline(x=0.12, color='black', ls='--', lw=2)
ax[1].set_xlabel('$\sigma_{treat,r}$')
ax[1].set_ylabel('Density')
ax[1].set_xlim([0.075,.33])
ax[1].legend(loc='upper right')
#%% Optional: Export als png
#fig.savefig("Figure3.png", format="png")
#%% ------------------------------------------------------------------------------
# Third Conditioning
# True H3: Sigma increases over rounds, and faster for the control group
# --------------------------------------------------------------------------------
coef_trueH3 = {
    'sigma_round':jnp.array([[.1, .17 ,.2],[.1 , .14, .18]]),
    'theta':jnp.array([[0, 0 , 0 ], [0, 0, 0]]), 
    'beta_treat':jnp.array([0,0],)
    }
#%% Generate synthetic data which aligns with these coefficient values
# Note: Use the dictionary without response data here, just pass on information about the data structure
condModel_H3 = numpyro.handlers.condition(model_partpool, data=coef_trueH3)
prior_predictive_Cond_H3 = Predictive(condModel_H3, num_samples=1)
prior_samples_Cond_H3 = prior_predictive_Cond_H3(random.PRNGKey(0), **dat_list_onlyIDs) 
#%% inspect whether the conditioned coefficient values where indeed maintained
print('theta', prior_samples_Cond_H3['theta'])
print('sigma_round', prior_samples_Cond_H3['sigma_round'][0,0])
print('beta_treat', prior_samples_Cond_H3['beta_treat'])
#%% Save the conditioned data into a dataframe to ease visualization
dfPrior_condH3 = Data.copy()
dfPrior_condH3['delta_WR'] = prior_samples_Cond_H3['delta_WR'][0,:]
dfPrior_condH3['delta_SS'] = prior_samples_Cond_H3['delta_SS'][0,:]
dfPrior_condH3['delta_BB'] = -(dfPrior_condH3['delta_WR'] + dfPrior_condH3['delta_SS'])
#%% Visualize the generated data
fig,ax = plt.subplots(layout='constrained')
ax.hist(dfPrior_condH3.loc[((dfPrior_condH3['r']==2)& dfPrior_condH3['treat']==1),'delta_PSR_share'].values.flatten(),bins=50, label='r2, treat',alpha = 0.5, color='darkgreen')
sb.kdeplot(dfPrior_condH3.loc[((dfPrior_condH3['r']==2)& dfPrior_condH3['treat']==1),'delta_PSR_share'].values.flatten(),alpha = 0.5, color='darkgreen')
ax.hist(dfPrior_condH3.loc[((dfPrior_condH3['r']==2)& dfPrior_condH3['treat']==0),'delta_PSR_share'].values.flatten(),bins=50, label='r2,control',alpha = 0.5, color='lightgreen')
sb.kdeplot(dfPrior_condH3.loc[((dfPrior_condH3['r']==2)& dfPrior_condH3['treat']==0),'delta_PSR_share'].values.flatten(),alpha = 0.5, color='lightgreen')
ax.set(xlabel='Value of delta_WR', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')
#%% Use the generated data to run the 'inference procedure' (passing it into the MCMC)
model_dummy_mcmc_H3 = MCMC(NUTS(model_partpool, init_strategy = init_to_median), num_warmup=1000, num_samples=1000, num_chains=4)
model_dummy_mcmc_H3.run(random.PRNGKey(0), 
    r=Data.r.values,
    treat=Data.treat.values,
    delta_WR = prior_samples_Cond_H3['delta_WR'][0,:], 
    delta_SS = prior_samples_Cond_H3['delta_SS'][0,:])
#%% Inspect the coefficient information to check whether it aligns with conditioned parameter values
mcmc_summary_dummy_H3 = model_dummy_mcmc_H3.print_summary(0.89)
#%% Draw from the posterior to generate a sample to be used for visualization
dummy_sample_H3_posteriorSample = model_dummy_mcmc_H3.get_samples(group_by_chain=True)
#%% Visualize the obtained parameter values in comparison to the pre-defined conditioned values
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
# Figure 4, left plot
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][0,:,0,0].flatten(), label='$\sigma_{control,0}$', color='darkgreen', alpha=0.7, ax=ax[0])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][1,:,0,0].flatten(),  color='darkgreen', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][2,:,0,0].flatten(),  color='darkgreen', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][3,:,0,0].flatten(),  color='darkgreen', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][0,:,0,1].flatten(), label='$\sigma_{control,1}$', color='darkred', alpha=0.7, ax=ax[0])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][1,:,0,1].flatten(),  color='darkred', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][2,:,0,1].flatten(),  color='darkred', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][3,:,0,1].flatten(),  color='darkred', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][0,:,0,2].flatten(), label='$\sigma_{control,2}$', color='darkblue', alpha=0.7, ax=ax[0])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][1,:,0,2].flatten(),  color='darkblue', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][2,:,0,2].flatten(),  color='darkblue', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][3,:,0,2].flatten(),  color='darkblue', alpha=0.5, ax=ax[0])
ax[0].set_xlabel('$\sigma_{control,r}$')
ax[0].set_ylabel('Density')
ax[0].axvline(x=0.2, color='black', ls='--', lw=2, label='Cond.Value')
ax[0].axvline(x=0.1, color='black', ls='--', lw=2)
ax[0].axvline(x=0.17, color='black', ls='--', lw=2)
ax[0].set_xlim([0.08,0.3])
ax[0].legend(loc='upper right')
# Figure 4, right plot
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][0,:,1,0].flatten(), label='$\sigma_{treat,0}$', color='lightgreen', alpha=0.7, ax=ax[1])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][1,:,1,0].flatten(),  color='lightgreen', alpha=0.5, ax=ax[1])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][2,:,1,0].flatten(),  color='lightgreen', alpha=0.5, ax=ax[1])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][3,:,1,0].flatten(),  color='lightgreen', alpha=0.5, ax=ax[1])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][0,:,1,1].flatten(), label='$\sigma_{treat,1}$', color='darkorange', alpha=0.7, ax=ax[1])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][1,:,1,1].flatten(),  color='darkorange', alpha=0.5, ax=ax[1])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][2,:,1,1].flatten(),  color='darkorange', alpha=0.5, ax=ax[1])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][3,:,1,1].flatten(),  color='darkorange', alpha=0.5, ax=ax[1])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][0,:,1,2].flatten(), label='$\sigma_{treat,2}$', color='lightblue', alpha=0.7, ax=ax[1])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][1,:,1,2].flatten(),  color='lightblue', alpha=0.5, ax=ax[1])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][2,:,1,2].flatten(),  color='lightblue', alpha=0.5, ax=ax[1])
sb.kdeplot(dummy_sample_H3_posteriorSample['sigma_round'][3,:,1,2].flatten(),  color='lightblue', alpha=0.5, ax=ax[1])
ax[1].set_xlabel('$\sigma_{treat,r}$')
ax[1].set_ylabel('Density')
ax[1].axvline(x=0.1, color='black', ls='--', lw=2, label='Cond.Value')
ax[1].axvline(x=0.14, color='black', ls='--', lw=2)
ax[1].axvline(x=0.18, color='black', ls='--', lw=2)
ax[1].set_xlim([0.08,0.3])
ax[1].legend(loc='upper right')
#%% Optional: Export als png
#fig.savefig("Figure4.png", format="png")
#%% ------------------------------------------------------------------------------
# Fourth Conditioning
# All Hypotheses are true AND there is a treatment effect
# --------------------------------------------------------------------------------
coef_trueallH = {
    'sigma_round':jnp.array([[.15, .12 ,.2],[.15 , .14, .18]]),
    'theta':jnp.array([[-0.2, 0 , 0 ], [-0.2, 0, 0]]), 
    'beta_treat':jnp.array([0.2, 0.2],)
    }
#%% Generate synthetic data which aligns with these coefficient values
# Note: Use the dictionary without response data here, just pass on information about the data structure
condModel_allH = numpyro.handlers.condition(model_partpool, data=coef_trueallH)
prior_predictive_Cond_allH = Predictive(condModel_allH, num_samples=1)
prior_samples_Cond_allH = prior_predictive_Cond_allH(random.PRNGKey(0), **dat_list_onlyIDs) 
#%% inspect whether the conditioned coefficient values where indeed maintained
print('theta', prior_samples_Cond_allH['theta'])
print('sigma_round', prior_samples_Cond_allH['sigma_round'][0,0])
print('beta_treat', prior_samples_Cond_allH['beta_treat'])
#%% Save the conditioned data into a dataframe to ease visualization
dfPrior_condallH = Data.copy()
dfPrior_condallH['delta_WR'] = prior_samples_Cond_allH['delta_WR'][0,:]
dfPrior_condallH['delta_SS'] = prior_samples_Cond_allH['delta_SS'][0,:]
dfPrior_condallH['delta_BB'] = -(dfPrior_condallH['delta_WR'] + dfPrior_condallH['delta_SS'])
#%% Use the generated data to run the 'inference procedure' (passing it into the MCMC)
model_dummy_mcmc_allH = MCMC(NUTS(model_partpool, init_strategy = init_to_median), num_warmup=1000, num_samples=1000, num_chains=4)
model_dummy_mcmc_allH.run(random.PRNGKey(0), 
    r=Data.r.values,
    treat=Data.treat.values,
    delta_WR = prior_samples_Cond_allH['delta_WR'][0,:], 
    delta_SS = prior_samples_Cond_allH['delta_SS'][0,:])
#%% Inspect the coefficient information to check whether it aligns with conditioned parameter values
mcmc_summary_dummy_allH = model_dummy_mcmc_allH.print_summary(0.89)
#%% Draw from the posterior to generate a sample to be used for visualization
dummy_sample_allH_posteriorSample = model_dummy_mcmc_allH.get_samples(group_by_chain=True)
#%% Visualize the obtained parameter values in comparison to the pre-defined conditioned values
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
# Figure 5, first row
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][1,:,0,0].flatten(), label='$\\theta_{WR,0}$', color='green', alpha=0.7, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][2,:,0,0].flatten(),  color='green', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][3,:,0,0].flatten(),  color='green', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][0,:,0,0].flatten(),  color='green', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][0,:,1,0].flatten(), label='$\\theta_{SS,0}$', color='purple', alpha=0.7, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][1,:,1,0].flatten(),  color='purple', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][2,:,1,0].flatten(),  color='purple', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][3,:,1,0].flatten(),  color='purple', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][1,:,0,1].flatten(), label='$\\theta_{WR,1}$', color='red', alpha=0.7, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][2,:,0,1].flatten(),  color='red', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][3,:,0,1].flatten(),  color='red', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][0,:,0,1].flatten(),  color='red', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][0,:,1,1].flatten(), label='$\\theta_{SS,1}$', color='orange', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][1,:,1,1].flatten(),  color='orange', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][2,:,1,1].flatten(),  color='orange', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][3,:,1,1].flatten(),  color='orange', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][1,:,0,2].flatten(), label='$\\theta_{WR,2}$', color='blue', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][2,:,0,2].flatten(),  color='blue', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][3,:,0,2].flatten(),  color='blue', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][0,:,0,2].flatten(),  color='blue', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][0,:,1,2].flatten(), label='$\\theta_{SS,2}$', color='grey', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][1,:,1,2].flatten(),  color='grey', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][2,:,1,2].flatten(),  color='grey', alpha=0.5, ax=ax[0,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['theta'][3,:,1,2].flatten(),  color='grey', alpha=0.5, ax=ax[0,0])
ax[0,0].axvline(x=-0.2, color='black', ls='--', lw=2, label='Cond. Value')
ax[0,0].axvline(x=0, color='black', ls='--', lw=2)
ax[0,0].set_xlabel('$\\theta_{r,tech}$')
ax[0,0].set_ylabel('Density')
ax[0,0].set_xlim([-0.27,0.25])
ax[0,0].legend(loc='upper right')
ax[0,1].axis('off')
# Figure 5, second row, left plot
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][0,:,0,0].flatten(), label='$\sigma_{control,0}$', color='darkgreen', alpha=0.7, ax=ax[1,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][1,:,0,0].flatten(),  color='darkgreen', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][2,:,0,0].flatten(),  color='darkgreen', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][3,:,0,0].flatten(),  color='darkgreen', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][0,:,0,1].flatten(), label='$\sigma_{control,1}$', color='darkred', alpha=0.7, ax=ax[1,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][1,:,0,1].flatten(),  color='darkred', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][2,:,0,1].flatten(),  color='darkred', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][3,:,0,1].flatten(),  color='darkred', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][0,:,0,2].flatten(), label='$\sigma_{control,2}$', color='darkblue', alpha=0.7, ax=ax[1,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][1,:,0,2].flatten(),  color='darkblue', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][2,:,0,2].flatten(),  color='darkblue', alpha=0.5, ax=ax[1,0])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][3,:,0,2].flatten(),  color='darkblue', alpha=0.5, ax=ax[1,0])
ax[1,0].axvline(x=0.15, color='black', ls='--', lw=2, label='Cond. Value')
ax[1,0].axvline(x=0.12, color='black', ls='--', lw=2)
ax[1,0].axvline(x=0.2, color='black', ls='--', lw=2)
ax[1,0].set_xlabel('$\sigma_{control,r}$')
ax[1,0].set_ylabel('Density')
ax[1,0].set_xlim([0.08,.3])
ax[1,0].legend(loc='upper right')
# Figure 5, second row, right plot
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][0,:,1,0].flatten(), label='$\sigma_{treat,0}$', color='lightgreen', ax=ax[1,1])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][1,:,1,0].flatten(),  color='lightgreen', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][2,:,1,0].flatten(),  color='lightgreen', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][3,:,1,0].flatten(),  color='lightgreen', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][0,:,1,1].flatten(), label='$\sigma_{treat,1}$', color='darkorange', ax=ax[1,1])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][1,:,1,1].flatten(),  color='darkorange', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][2,:,1,1].flatten(),  color='darkorange', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][3,:,1,1].flatten(),  color='darkorange', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][0,:,1,2].flatten(), label='$\sigma_{treat,2}$', color='lightblue', ax=ax[1,1])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][1,:,1,2].flatten(),  color='lightblue', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][2,:,1,2].flatten(),  color='lightblue', alpha=0.5, ax=ax[1,1])
sb.kdeplot(dummy_sample_allH_posteriorSample['sigma_round'][3,:,1,2].flatten(),  color='lightblue', alpha=0.5, ax=ax[1,1])
ax[1,1].axvline(x=0.15, color='black', ls='--', lw=2, label='Cond. Value')
ax[1,1].axvline(x=0.14, color='black', ls='--', lw=2)
ax[1,1].axvline(x=0.18, color='black', ls='--', lw=2)
ax[1,1].set_xlabel('$\sigma_{treat,r}$')
ax[1,1].set_ylabel('Density')
ax[1,1].set_xlim([0.08,.3])
ax[1,1].legend(loc='upper right')
#%% Optional: Export als png
#fig.savefig("Figure5.png", format="png")
#%% ------------------------------------------------------------------------------
# Comparison of our model and a no-thetaects-model for estimating the treatment effect
#---------------------------------------------------------------------------------
#%% Define a model without group-specific variance or switching parameters
def model_notheta(treat, delta_WR=None, delta_SS=None): 
    
    # set a prior for the treatment effect, same as above
    beta_treat = numpyro.sample("beta_treat", dist.Normal(0, 0.3).expand([2]))

    mu_delta_WR = beta_treat[0]*treat
    mu_delta_SS = beta_treat[1]*treat

    # set a static prior for the variance term
    sigma = numpyro.sample('sigma', dist.Exponential(1))

    # draw posterior distributions
    delta_WR = numpyro.sample('delta_WR', dist.TruncatedNormal(loc=mu_delta_WR, scale=sigma,low=-1, high=1), obs=delta_WR)
    delta_SS = numpyro.sample('delta_SS', dist.TruncatedNormal(loc=mu_delta_SS, scale=sigma,low=-1, high=1), obs=delta_SS)
#%% Passing the synthetic data generated above ('all hypotheses are true'-conditioning) into the model without round effects    
model_dummy_mcmc_noRound = MCMC(NUTS(model_notheta, init_strategy = init_to_median), num_warmup=1000, num_samples=1000, num_chains=4)
model_dummy_mcmc_noRound.run(random.PRNGKey(0), 
    treat=Data.treat.values,
    delta_WR = prior_samples_Cond_allH['delta_WR'][0,:], 
    delta_SS = prior_samples_Cond_allH['delta_SS'][0,:])

mcmc_summary = model_dummy_mcmc_noRound.print_summary(0.89)
#%% draw a sample from the posteriors obtained from the model without round effects
samples_mcmc_noRound = model_dummy_mcmc_noRound.get_samples(group_by_chain=True)
#%% Visualization of the treatment effect coefficient in comparison the the conditioned value
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
# Used in Figure 6, left plot
sb.kdeplot(dummy_sample_allH_posteriorSample['beta_treat'][0,:,0].flatten(), label='$\\beta_{WR}$', color='orange', alpha=0.7, ax=ax[0])
sb.kdeplot(dummy_sample_allH_posteriorSample['beta_treat'][1,:,0].flatten(),  color='orange', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_allH_posteriorSample['beta_treat'][2,:,0].flatten(),  color='orange', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_allH_posteriorSample['beta_treat'][3,:,0].flatten(),  color='orange', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_allH_posteriorSample['beta_treat'][0,:,1].flatten(), label='$\\beta_{SS}$', color='purple', alpha=0.7, ax=ax[0])
sb.kdeplot(dummy_sample_allH_posteriorSample['beta_treat'][1,:,1].flatten(),  color='purple', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_allH_posteriorSample['beta_treat'][2,:,1].flatten(),  color='purple', alpha=0.5, ax=ax[0])
sb.kdeplot(dummy_sample_allH_posteriorSample['beta_treat'][3,:,1].flatten(),  color='purple', alpha=0.5, ax=ax[0])
ax[0].set_xlabel('$\\beta_{tech}$')  
ax[0].axvline(x=0.2, color='black', ls='--', lw=2, label='Cond.Value')
ax[0].set_ylabel('Density')
ax[0].set_xlim([0.1,0.3])
ax[0].legend(loc='upper right')
# Visualization of the treatment effect coefficient obtained from the model without round effects
# Used in Figure 6, right plot
sb.kdeplot(samples_mcmc_noRound['beta_treat'][0,:,0].flatten(), label='$\\beta_{WR}$', color='orange', alpha=0.7, ax=ax[1])
sb.kdeplot(samples_mcmc_noRound['beta_treat'][1,:,0].flatten(),  color='orange', alpha=0.5, ax=ax[1])
sb.kdeplot(samples_mcmc_noRound['beta_treat'][2,:,0].flatten(),  color='orange', alpha=0.5, ax=ax[1])
sb.kdeplot(samples_mcmc_noRound['beta_treat'][3,:,0].flatten(),  color='orange', alpha=0.5, ax=ax[1])
sb.kdeplot(samples_mcmc_noRound['beta_treat'][0,:,1].flatten(), label='$\\beta_{SS}$', color='purple', alpha=0.7, ax=ax[1])
sb.kdeplot(samples_mcmc_noRound['beta_treat'][1,:,1].flatten(),  color='purple', alpha=0.5, ax=ax[1])
sb.kdeplot(samples_mcmc_noRound['beta_treat'][2,:,1].flatten(),  color='purple', alpha=0.5, ax=ax[1])
sb.kdeplot(samples_mcmc_noRound['beta_treat'][3,:,1].flatten(),  color='purple', alpha=0.5, ax=ax[1])
ax[1].set_xlabel('$\\beta_{tech}$')  
ax[1].axvline(x=0.2, color='black', ls='--', lw=2, label='Cond.Value')
ax[1].set_ylabel('Density')
ax[1].set_xlim([0,0.3])
ax[1].legend(loc='upper right')
#%% Optional: Export als png
#fig.savefig("Figure6.png", format="png")
#%% Visualization of the sigma parameter obtained from the model without round effects
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(samples_mcmc_noRound['sigma'][0,:].flatten(), label='$\sigma$', color='grey', alpha=0.7)
sb.kdeplot(samples_mcmc_noRound['sigma'][1,:].flatten(),  color='grey', alpha=0.5)
sb.kdeplot(samples_mcmc_noRound['sigma'][2,:].flatten(),  color='grey', alpha=0.5)
sb.kdeplot(samples_mcmc_noRound['sigma'][3,:].flatten(),  color='grey', alpha=0.5)
ax.set_xlabel('$\sigma$')  
ax.set_ylabel('Density')
ax.legend(loc='upper right')
#%% -----------------------------------------------------------------------------
#-------RUN MODEL WITH REAL Experiment DATA -------------------------------------
#--------------------------------------------------------------------------------
#%% Run the inference procedure using the dictionary containing the real data
model_partpool_realDat = MCMC(NUTS(model_partpool, init_strategy = init_to_median), num_warmup=1000, 
num_samples=1000, num_chains=4)
model_partpool_realDat.run(random.PRNGKey(0), **dat_list_allOutputs)
#%% Exensive estimated coefficient information 
# Table S4 in the Supplementary Material
mcmc_summary_partpool = model_partpool_realDat.print_summary(0.89)
#%% Draw a sample from the posterior to be used for visualization
samples_partpool_realDat = model_partpool_realDat.get_samples(group_by_chain=True)
#%% Traceplot for chain inspection and parameter estimate overview
# Figure S1 in the Supplementary Material
az.plot_trace(samples_partpool_realDat)
# Export as png if needed
#plt.savefig("FigureS1.png", format="png")
#%% -----------------------------------------------------------------------------
# Visualize the obtained parameter values
# -----------------------------------------------------------------------------
#%% Visualization of the theta coefficient
# Figure 7
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(samples_partpool_realDat['theta'][1,:,0,0].flatten(), label='$\\theta_{WR,0}$', color='green', alpha=0.7)
sb.kdeplot(samples_partpool_realDat['theta'][2,:,0,0].flatten(),  color='green', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][3,:,0,0].flatten(),  color='green', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][0,:,0,0].flatten(),  color='green', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][0,:,1,0].flatten(), label='$\\theta_{SS,0}$', color='purple', alpha=0.7)
sb.kdeplot(samples_partpool_realDat['theta'][1,:,1,0].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][2,:,1,0].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][3,:,1,0].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][1,:,0,1].flatten(), label='$\\theta_{WR,1}$', color='red', alpha=0.7)
sb.kdeplot(samples_partpool_realDat['theta'][2,:,0,1].flatten(),  color='red', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][3,:,0,1].flatten(),  color='red', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][0,:,0,1].flatten(),  color='red', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][0,:,1,1].flatten(), label='$\\theta_{SS,1}$', color='orange', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][1,:,1,1].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][2,:,1,1].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][3,:,1,1].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][1,:,0,2].flatten(), label='$\\theta_{WR,2}$', color='blue', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][2,:,0,2].flatten(),  color='blue', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][3,:,0,2].flatten(),  color='blue', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][0,:,0,2].flatten(),  color='blue', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][0,:,1,2].flatten(), label='$\\theta_{SS,2}$', color='grey', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][1,:,1,2].flatten(),  color='grey', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][2,:,1,2].flatten(),  color='grey', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['theta'][3,:,1,2].flatten(),  color='grey', alpha=0.5)
ax.set_xlabel('$\\theta_{r,tech}$')
ax.set_ylabel('Density')
ax.set_xlim([-0.15,0.2])
ax.legend(loc='upper right')
# Export as png if needed
#fig.savefig("Figure7.png", format="png")
#%% Visualization of the sigma coefficient
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
# Figure 8, left plot
sb.kdeplot(samples_partpool_realDat['sigma_round'][0,:,0,0].flatten(), label='$\sigma_{control,0}$', color='darkgreen', alpha=0.7, ax=ax[0])
sb.kdeplot(samples_partpool_realDat['sigma_round'][1,:,0,0].flatten(),  color='darkgreen', alpha=0.5, ax=ax[0])
sb.kdeplot(samples_partpool_realDat['sigma_round'][2,:,0,0].flatten(),  color='darkgreen', alpha=0.5, ax=ax[0])
sb.kdeplot(samples_partpool_realDat['sigma_round'][3,:,0,0].flatten(),  color='darkgreen', alpha=0.5, ax=ax[0])
sb.kdeplot(samples_partpool_realDat['sigma_round'][0,:,0,1].flatten(), label='$\sigma_{control,1}$', color='darkred', alpha=0.7, ax=ax[0])
sb.kdeplot(samples_partpool_realDat['sigma_round'][1,:,0,1].flatten(),  color='darkred', alpha=0.5, ax=ax[0])
sb.kdeplot(samples_partpool_realDat['sigma_round'][2,:,0,1].flatten(),  color='darkred', alpha=0.5, ax=ax[0])
sb.kdeplot(samples_partpool_realDat['sigma_round'][3,:,0,1].flatten(),  color='darkred', alpha=0.5, ax=ax[0])
sb.kdeplot(samples_partpool_realDat['sigma_round'][0,:,0,2].flatten(), label='$\sigma_{control,2}$', color='darkblue', alpha=0.7, ax=ax[0])
sb.kdeplot(samples_partpool_realDat['sigma_round'][1,:,0,2].flatten(),  color='darkblue', alpha=0.5, ax=ax[0])
sb.kdeplot(samples_partpool_realDat['sigma_round'][2,:,0,2].flatten(),  color='darkblue', alpha=0.5, ax=ax[0])
sb.kdeplot(samples_partpool_realDat['sigma_round'][3,:,0,2].flatten(),  color='darkblue', alpha=0.5, ax=ax[0])
ax[0].set_xlabel('$\sigma_{control,r}$')
ax[0].set_ylabel('Density')
ax[0].set_xlim([0.1,0.4])
ax[0].legend(loc='upper right')
# Figure 8, right plot
sb.kdeplot(samples_partpool_realDat['sigma_round'][0,:,1,0].flatten(), label='$\sigma_{treat,0}$', color='lightgreen', ax=ax[1])
sb.kdeplot(samples_partpool_realDat['sigma_round'][1,:,1,0].flatten(),  color='lightgreen', alpha=0.5, ax=ax[1])
sb.kdeplot(samples_partpool_realDat['sigma_round'][2,:,1,0].flatten(),  color='lightgreen', alpha=0.5, ax=ax[1])
sb.kdeplot(samples_partpool_realDat['sigma_round'][3,:,1,0].flatten(),  color='lightgreen', alpha=0.5, ax=ax[1])
sb.kdeplot(samples_partpool_realDat['sigma_round'][0,:,1,1].flatten(), label='$\sigma_{treat,1}$', color='darkorange', ax=ax[1])
sb.kdeplot(samples_partpool_realDat['sigma_round'][1,:,1,1].flatten(),  color='darkorange', alpha=0.5, ax=ax[1])
sb.kdeplot(samples_partpool_realDat['sigma_round'][2,:,1,1].flatten(),  color='darkorange', alpha=0.5, ax=ax[1])
sb.kdeplot(samples_partpool_realDat['sigma_round'][3,:,1,1].flatten(),  color='darkorange', alpha=0.5, ax=ax[1])
sb.kdeplot(samples_partpool_realDat['sigma_round'][0,:,1,2].flatten(), label='$\sigma_{treat,2}$', color='lightblue', ax=ax[1])
sb.kdeplot(samples_partpool_realDat['sigma_round'][1,:,1,2].flatten(),  color='lightblue', alpha=0.5, ax=ax[1])
sb.kdeplot(samples_partpool_realDat['sigma_round'][2,:,1,2].flatten(),  color='lightblue', alpha=0.5, ax=ax[1])
sb.kdeplot(samples_partpool_realDat['sigma_round'][3,:,1,2].flatten(),  color='lightblue', alpha=0.5, ax=ax[1])
ax[1].set_xlabel('$\sigma_{treat,r}$')
ax[1].set_ylabel('Density')
ax[1].set_xlim([0.1,0.4])
ax[1].legend(loc='upper right')
#%% Export as png if needed
fig.savefig("Figure8.png", format="png")
#%% Visualization of the treatment effect coefficient
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(samples_partpool_realDat['beta_treat'][0,:,0].flatten(), label='$\\beta_{WR}$', color='orange', alpha=0.7)
sb.kdeplot(samples_partpool_realDat['beta_treat'][1,:,0].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['beta_treat'][2,:,0].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['beta_treat'][3,:,0].flatten(),  color='orange', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['beta_treat'][0,:,1].flatten(), label='$\\beta_{SS}$', color='purple', alpha=0.7)
sb.kdeplot(samples_partpool_realDat['beta_treat'][1,:,1].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['beta_treat'][2,:,1].flatten(),  color='purple', alpha=0.5)
sb.kdeplot(samples_partpool_realDat['beta_treat'][3,:,1].flatten(),  color='purple', alpha=0.5)
ax.set_xlabel('$\\beta_{tech}$')  
ax.set_ylabel('Density')
ax.set_xlim([-0.1,0.15])
ax.legend(loc='upper right')
