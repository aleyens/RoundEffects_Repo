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

if "SVG" in os.environ:
    %config InlineBackend.figure_formats = ["svg"]
warnings.formatwarning = lambda message, category, *args, **kwargs: "{}: {}\n".format(
    category.__name__, message
)
az.style.use("arviz-darkgrid")
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)

#%% ----------------------
# Data loading and preparation
#-------------------------
realData = pd.read_excel(r'N:\ds\project\TrAgS\Alexa\Data_Models_Alexa\RoundEffects\RoundEffect_Data_clean_long_22102023_R_AL.xlsx')
realData = realData.dropna()
realData['ID_Neu'] = realData['ID_Neu'].astype('int64')
len(realData['ID_Neu'].unique())

#%% #select only the needed columns
d = realData[[ 'ID_Neu' , 'round' , 'treatment', 'PSR_shifted', 'SS_shifted', 'BB' ]]
#%%
d['treat'] = d['treatment'].astype('int16')
#%%
len(d.loc[(d['round']==3)&(d['treat']==0)])
#%%transform PSR variable
d['PSR_shifted'] = d['PSR_shifted'].astype('int16')
d['delta_PSR'] = d['PSR_shifted'] - d.sort_values(by=['round']).groupby(by=['ID_Neu'])['PSR_shifted'].shift(1)
#%% same transformation for BB and SS variables as well 
d['SS_shifted'] = d['SS_shifted'].astype('int16')
d['delta_SS'] = d['SS_shifted'] - d.sort_values(by=['round']).groupby(by=['ID_Neu'])['SS_shifted'].shift(1)
#%%
d['BB'] = d['BB'].astype('int16')
#%%
d['delta_BB'] = d['BB'] - d.sort_values(by=['round']).groupby(by=['ID_Neu'])['BB'].shift(1)
d
#%% re-define 'normal' df as all rounds but the 0
d_not0 = d[d['round'] != 1]
#%%
d_not0['r'] = d_not0['round']-2
d_not0['r'] = d_not0['r'].astype('int16')
d_not0
#%%
len(d_not0.loc[(d_not0['r']==1)&(d_not0['treat']==1)])
#%%
d_not0.groupby(by=['r', 'treat']).count()
#%% compute all outputs as shares
d_not0['delta_PSR_share'] = (d_not0['delta_PSR']/50)
d_not0['delta_SS_share'] = (d_not0['delta_SS']/50)
d_not0['delta_BB_share'] = (d_not0['delta_BB']/50)


#%% real dat visualization
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(d_not0.loc[((d_not0['r']==2)& d_not0['treat']==0),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='green')
sb.kdeplot(d_not0.loc[((d_not0['r']==1)& d_not0['treat']==0),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='red')
sb.kdeplot(d_not0.loc[((d_not0['r']==0)& d_not0['treat']==0),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='blue')
ax.set(xlabel='Value of delta_WR', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')
#%%
fig,ax = plt.subplots(layout='constrained')
#ax.hist(dfPrior.loc[((d_not0['r']==2) & d_not0['treat']==1),'delta_PSR_share'].values.flatten(),bins=50, label='r2, treat',alpha = 0.5, color='green')
sb.kdeplot(d_not0.loc[((d_not0['r']==2)& d_not0['treat']==1),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='green')
#ax.hist(dfPrior.loc[((dfPrior['r']==1)& d_not0['treat']==1),'delta_PSR_share'].values.flatten(),bins=50, label='r1, treat',alpha = 0.5, color='red')
sb.kdeplot(d_not0.loc[((d_not0['r']==1)& d_not0['treat']==1),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='red')
#ax.hist(dfPrior.loc[((dfPrior['r']==0)& d_not0['treat']==1),'delta_PSR_share'].values.flatten(),bins=50, label='r0, treat',alpha = 0.3, color='blue')
sb.kdeplot(d_not0.loc[((d_not0['r']==0)& d_not0['treat']==1),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='blue')
ax.set(xlabel='Value of delta_WR', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')
#%%
d_not0.r.values,
#d_not0.treat.values

#%%
#struct = pd.read_excel(r'N:\ds\project\TrAgS\Alexa\Data_Models_Alexa\RoundEffects\roundeff_struktur_IDs.xlsx')
#struct.dropna()
#%%
#struct['r'] = (struct['round']-1).astype('int16')
#struct['treat'] = struct['treatment'].astype('int16')
#%%
dat_list_onlyIDs = dict(
    r=d_not0.r.values,
    treat=d_not0.treat.values
)
dat_list_onlyIDs
#%%create dictionary with real data in shares per output
dat_list_allOutputs_shares = dict(
    r=d_not0.r.values,
    treat=d_not0.treat.values,
    delta_PSR_share = d_not0.delta_PSR_share.values,
    delta_SS_share = d_not0.delta_SS_share.values,
    #delta_BB = d_not0.delta_BB_share.values
)
d_not0.r.values
#%% ------------------------------------------------------------------------------
# Partially pooled model with varying sigma and roundeff per treatment group
# --------------------------------------------------------------------------------
#%%
#includes meta-level where sigma priors depend on the treatment group

def model_partpool(r, treat, delta_PSR_share=None, delta_SS_share=None): 
    
     #round effect 
    roundeff = numpyro.sample("roundeff", dist.Normal(0, 0.3), sample_shape=(2,3)) #per tech and round

    # variance per round
    sigma_bar = numpyro.sample('sigma_bar', dist.TruncatedNormal(loc=1, scale=0.2, low=0))
    sigma_round = numpyro.sample("sigma_round", dist.Exponential(sigma_bar), sample_shape=(2,3)) #per round and treat

    btreat = numpyro.sample("btreat", dist.Normal(0, 0.3).expand([2])) #per tech

    mu_delta_PSR = roundeff[0,r] + btreat[0]*treat
    mu_delta_SS = roundeff[1,r] + btreat[1]*treat

    sigma = sigma_round[treat, r]

    delta_PSR_share = numpyro.sample('delta_PSR_share', dist.TruncatedNormal(loc=mu_delta_PSR, scale=sigma,low=-1, high=1), obs=delta_PSR_share)
    delta_SS_share = numpyro.sample('delta_SS_share', dist.TruncatedNormal(loc=mu_delta_SS, scale=sigma,low=-1, high=1), obs=delta_SS_share)

#%% -------------------
# Now condition the dummy generation to specific values and see whether they some through in the mcmc
# ---------------------
#%% --------------------------------
# First : True H1 - so the roundeff in r=0 (when moving from r0 in r1 of the experiment) is non-zero
#-----------------------------------

coef_trueH1_partpool = {
    'sigma_round':jnp.array([[.2, .2 ,.2],[.2 , .2, .2 ]]),
    'roundeff':jnp.array([[-0.2,0,0],[-0.2,0,0]]),
    'btreat':jnp.array([0.,0.],)
    }

#%% Generate synthetic data which proves this 
partpool_condModel_H1 = numpyro.handlers.condition(model_partpool, data=coef_trueH1_partpool)
prior_predictive_Partpool_Cond_H1 = Predictive(partpool_condModel_H1, num_samples=1)
#%%
prior_samples_Partpool_Cond_H1 = prior_predictive_Partpool_Cond_H1(random.PRNGKey(0), **dat_list_onlyIDs)
#%%
print('roundeff', prior_samples_Partpool_Cond_H1['roundeff'])
print('sigma_round', prior_samples_Partpool_Cond_H1['sigma_round'][0,0])
print('btreat', prior_samples_Partpool_Cond_H1['btreat'])
#%% save them into new df
dfPrior = d_not0.copy()
dfPrior['delta_PSR_share'] = prior_samples_Partpool_Cond_H1['delta_PSR_share'][0,:]
dfPrior['delta_SS_share'] = prior_samples_Partpool_Cond_H1['delta_SS_share'][0,:]
dfPrior['delta_BB_share'] = -(dfPrior['delta_PSR_share'] + dfPrior['delta_SS_share'])
#%% plot sampled PSR for the different groups
fig,ax = plt.subplots(layout='constrained')
#ax.hist(dfPrior.loc[(dfPrior['r']==2) ,'delta_PSR_share'].values.flatten(),bins=50, label='r2',alpha = 0.5, color='green')
#sb.kdeplot(dfPrior.loc[((dfPrior['r']==2)),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='green')
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
#ax.hist(dfPrior.loc[((dfPrior['r']==0)),'delta_SS_share'].values.flatten(),bins=50, label='r0',alpha = 0.3, color='blue')
#sb.kdeplot(dfPrior.loc[((dfPrior['r']==0)),'delta_SS_share'].values.flatten(), alpha = 0.5, color='blue')
ax.set(xlabel='Value of delta_SS', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')
#%% run MCMC with this to check
model_dummy_mcmc_H1_partpool = MCMC(NUTS(model_partpool, init_strategy = init_to_median), num_warmup=1000, num_samples=1000, num_chains=4)
model_dummy_mcmc_H1_partpool.run(random.PRNGKey(0), r=d_not0.r.values,
    treat=d_not0.treat.values,delta_PSR_share = prior_samples_Partpool_Cond_H1['delta_PSR_share'][0,:], 
    delta_SS_share = prior_samples_Partpool_Cond_H1['delta_SS_share'][0,:])

mcmc_summary_dummy_H1_partpool = model_dummy_mcmc_H1_partpool.print_summary(0.89)
#%% get sample to use in plotting below
dummy_sample_H1_partpool = model_dummy_mcmc_H1_partpool.get_samples(group_by_chain=True)
dummy_sample_H1_partpool['roundeff'].shape

#%% generate posteriors for H1
#erste dimension: chain, zweite dimension: sample draw (always : to include all 1000), third demension: technology, 
# last dimesnion: round - [chain, : , tech, round]
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

#%% sigma plotting 
# first dimension: chain, secpnd dimension: sample draw (always : to include all 1000), third demension: treat vs control, 
# forth dimension: round - [chain, : , treat, round]
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
prior_samples_Partpool_Cond_H2 = prior_predictive_Partpool_Cond_H2(random.PRNGKey(0), **dat_list_onlyIDs)
#%%
print('sigma_round', prior_samples_Partpool_Cond_H2['sigma_round'].shape)
print('btreat', prior_samples_Partpool_Cond_H2['btreat'])
print('shape of delta_PSR:', prior_samples_Partpool_Cond_H2['delta_PSR_share'].shape)
#%% save them into new df
dfPrior = d_not0.copy()
dfPrior['delta_PSR_share'] = prior_samples_Partpool_Cond_H2['delta_PSR_share'][0,:]
dfPrior['delta_SS_share'] = prior_samples_Partpool_Cond_H2['delta_SS_share'][0,:]
dfPrior['delta_BB_share'] = -(dfPrior['delta_PSR_share'] + dfPrior['delta_SS_share'])

#%% plot sampled PSR for the different groups
fig,ax = plt.subplots(layout='constrained')
ax.hist(dfPrior.loc[((dfPrior['r']==2) & dfPrior['treat']==1),'delta_PSR_share'].values.flatten(),bins=50, label='r2',alpha = 0.5, color='green')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==2)& dfPrior['treat']==1),'delta_PSR_share'].values.flatten(),alpha = 0.5, color='green')
ax.hist(dfPrior.loc[((dfPrior['r']==1)& dfPrior['treat']==1),'delta_PSR_share'].values.flatten(),bins=50, label='r1',alpha = 0.5, color='red')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==1)& dfPrior['treat']==1),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='red')
ax.hist(dfPrior.loc[((dfPrior['r']==0)& dfPrior['treat']==1),'delta_PSR_share'].values.flatten(),bins=50, label='r0',alpha = 0.3, color='blue')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==0)& dfPrior['treat']==1),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='blue')
ax.set(xlabel='Value of delta_WR', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')
#%% 
fig,ax = plt.subplots(layout='constrained')
ax.hist(dfPrior.loc[((dfPrior['r']==2) & dfPrior['treat']==0),'delta_PSR_share'].values.flatten(),bins=50, label='r2',alpha = 0.5, color='green')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==2)& dfPrior['treat']==0),'delta_PSR_share'].values.flatten(),alpha = 0.5, color='green')
ax.hist(dfPrior.loc[((dfPrior['r']==1)& dfPrior['treat']==0),'delta_PSR_share'].values.flatten(),bins=50, label='r1',alpha = 0.5, color='red')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==1)& dfPrior['treat']==0),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='red')
ax.hist(dfPrior.loc[((dfPrior['r']==0)& dfPrior['treat']==0),'delta_PSR_share'].values.flatten(),bins=50, label='r0',alpha = 0.3, color='blue')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==0)& dfPrior['treat']==0),'delta_PSR_share'].values.flatten(), alpha = 0.5, color='blue')
ax.set(xlabel='Value of delta_WR', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')

#%% run MCMC with this to check
model_dummy_mcmc_H2_partpool = MCMC(NUTS(model_partpool, init_strategy = init_to_mean), num_warmup=2000, num_samples=2000, num_chains=4)
model_dummy_mcmc_H2_partpool.run(random.PRNGKey(0), r=d_not0.r.values,
    treat=d_not0.treat.values,
    delta_PSR_share = prior_samples_Partpool_Cond_H2['delta_PSR_share'][0,:], 
    delta_SS_share = prior_samples_Partpool_Cond_H2['delta_SS_share'][0,:])

mcmc_summary_dummy_H2_partpool = model_dummy_mcmc_H2_partpool.print_summary(0.89)
#%% get sample to use in plotting below
dummy_sample_H2_partpool = model_dummy_mcmc_H2_partpool.get_samples(group_by_chain=True)
dummy_sample_H2_partpool['sigma_round'].shape
#%% sigma plotting 
# first dimension: chain, secpnd dimension: sample draw (always : to include all 1000), third demension: treat vs control, 
# forth dimension: round - [chain, : , treat, round]
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

#%%
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
prior_samples_Partpool_Cond_H3 = prior_predictive_Partpool_Cond_H3(random.PRNGKey(0), **dat_list_onlyIDs)
#%%
print('roundeff', prior_samples_Partpool_Cond_H3['roundeff'])
print('sigma_round', prior_samples_Partpool_Cond_H3['sigma_round'])
print('btreat', prior_samples_Partpool_Cond_H3['btreat'])
print('shape of delta_PSR:', prior_samples_Partpool_Cond_H3['delta_PSR_share'].shape)
#%% save them into new df
dfPrior = d_not0.copy()
dfPrior['delta_PSR_share'] = prior_samples_Partpool_Cond_H3['delta_PSR_share'][0,:]
dfPrior['delta_SS_share'] = prior_samples_Partpool_Cond_H3['delta_SS_share'][0,:]
dfPrior['delta_BB_share'] = -(dfPrior['delta_PSR_share'] + dfPrior['delta_SS_share'])

#%% plot sampled PSR for the different groups
fig,ax = plt.subplots(layout='constrained')
ax.hist(dfPrior.loc[((dfPrior['r']==2)& dfPrior['treat']==1),'delta_PSR_share'].values.flatten(),bins=50, label='r2, treat',alpha = 0.5, color='darkgreen')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==2)& dfPrior['treat']==1),'delta_PSR_share'].values.flatten(),alpha = 0.5, color='darkgreen')
#ax.hist(dfPrior.loc[((dfPrior['r']==1)& dfPrior['treat']==1),'delta_PSR'].values.flatten(),bins=50, label='r1, treat',alpha = 0.5, color='red')
#sb.kdeplot(dfPrior.loc[((dfPrior['r']==1)& dfPrior['treat']==1),'delta_PSR'].values.flatten(),alpha = 0.5, color='red')
#ax.hist(dfPrior.loc[((dfPrior['r']==0)& dfPrior['treat']==1),'delta_PSR'].values.flatten(),bins=50, label='r0, treat',alpha = 0.3, color='darkblue')
#sb.kdeplot(dfPrior.loc[((dfPrior['r']==0)& dfPrior['treat']==1),'delta_PSR'].values.flatten(), alpha = 0.5, color='darkblue')

ax.hist(dfPrior.loc[((dfPrior['r']==2)& dfPrior['treat']==0),'delta_PSR_share'].values.flatten(),bins=50, label='r2,control',alpha = 0.5, color='lightgreen')
sb.kdeplot(dfPrior.loc[((dfPrior['r']==2)& dfPrior['treat']==0),'delta_PSR_share'].values.flatten(),alpha = 0.5, color='lightgreen')
#ax.hist(dfPrior.loc[((dfPrior['r']==1)& dfPrior['treat']==0),'delta_PSR'].values.flatten(),bins=50, label='r1,control',alpha = 0.5, color='orange')
#sb.kdeplot(dfPrior.loc[((dfPrior['r']==1)& dfPrior['treat']==0),'delta_PSR'].values.flatten(),alpha = 0.5, color='orange')
#ax.hist(dfPrior.loc[((dfPrior['r']==0)& dfPrior['treat']==0),'delta_PSR'].values.flatten(),bins=50, label='r0,control',alpha = 0.3, color='lightblue')
#sb.kdeplot(dfPrior.loc[((dfPrior['r']==0)& dfPrior['treat']==0),'delta_PSR'].values.flatten(), alpha = 0.5, color='lightblue')

ax.set(xlabel='Value of delta_WR', ylabel='Frequency in the dummy data')
ax.legend(loc='upper right')

#%% run MCMC with this to check
model_dummy_mcmc_H3_partpool = MCMC(NUTS(model_partpool, init_strategy = init_to_mean), num_warmup=1000, num_samples=1000, num_chains=4)
model_dummy_mcmc_H3_partpool.run(random.PRNGKey(0), r=d_not0.r.values,
    treat=d_not0.treat.values,
    delta_PSR_share = prior_samples_Partpool_Cond_H3['delta_PSR_share'][0,:], 
    delta_SS_share = prior_samples_Partpool_Cond_H3['delta_SS_share'][0,:])

mcmc_summary_dummy_H3_partpool = model_dummy_mcmc_H3_partpool.print_summary(0.89)
#works with positive, negative and 0 values for roundeffects!
#%% get sample to use in plotting below
dummy_sample_H3_partpool = model_dummy_mcmc_H3_partpool.get_samples(group_by_chain=True)
dummy_sample_H3_partpool['sigma_round'].shape
#%% plot sigma
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

#%% plot sigma
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
prior_samples_Partpool_Cond_allH = prior_predictive_Partpool_Cond_allH(random.PRNGKey(0), **dat_list_onlyIDs)
#%% run MCMC with this to check
model_dummy_mcmc_allH_partpool = MCMC(NUTS(model_partpool, init_strategy = init_to_median), num_warmup=1000, num_samples=1000, num_chains=4)
model_dummy_mcmc_allH_partpool.run(random.PRNGKey(0), r=d_not0.r.values,
    treat=d_not0.treat.values,
    delta_PSR_share = prior_samples_Partpool_Cond_allH['delta_PSR_share'][0,:], 
    delta_SS_share = prior_samples_Partpool_Cond_allH['delta_SS_share'][0,:])

mcmc_summary_dummy_allH_partpool = model_dummy_mcmc_allH_partpool.print_summary(0.89)

#%% get sample to use in plotting below
dummy_sample_allH_partpool = model_dummy_mcmc_allH_partpool.get_samples(group_by_chain=True)
dummy_sample_allH_partpool['roundeff'].shape

#%% generate posteriors for H1
fig,ax = plt.subplots(layout='constrained')
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][1,:,0,0].flatten(), label='$\\theta_{WR,0}$', color='green', alpha=0.7)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][2,:,0,0].flatten(),  color='green', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][3,:,0,0].flatten(),  color='green', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][0,:,0,0].flatten(),  color='green', alpha=0.5)
sb.kdeplot(dummy_sample_allH_partpool['roundeff'][0,:,1,0].flatten(), label='$\\theta_{SS,0}$', color='purple', alpha=0.7)
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

#%% Sigma
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

#%% Sigma
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



#%% treateff posterior for no model WITH roundeffs
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
    treat=d_not0.treat.values,
    delta_PSR_share = prior_samples_Partpool_Cond_allH['delta_PSR_share'][0,:], 
    delta_SS_share = prior_samples_Partpool_Cond_allH['delta_SS_share'][0,:])

mcmc_summary = model_dummy_mcmc_noRound.print_summary(0.89)

#%% compare posteriors of treateff for both models
samples_mcmc_noRound = model_dummy_mcmc_noRound.get_samples(group_by_chain=True)
samples_mcmc_noRound['sigma'].shape
#%% treateff posterior for no roundeff model
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

model_partpool_realDat = MCMC(NUTS(model_partpool, init_strategy = init_to_median), num_warmup=1000, 
num_samples=1000, num_chains=4)
model_partpool_realDat.run(random.PRNGKey(0), **dat_list_allOutputs_shares)

mcmc_summary_partpool = model_partpool_realDat.print_summary(0.89)

#%%  
samples_partpool_realDat = model_partpool_realDat.get_samples(group_by_chain=True)

#%% traceplot for chain inspection and parameter estimate overview
az.plot_trace(samples_partpool_realDat)

#%% plot roundeff parameter
#%% generate posteriors for H1
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
ax.set_xlim([0.1,0.4])
ax.legend(loc='upper right')

#%% Sigma
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
ax.set_xlim([0.1,0.4])
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
ax.set_xlim([-0.1,0.15])
ax.legend(loc='upper right')
