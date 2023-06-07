import numpy 
import h5py
import mne
import scipy.io
import scipy.signal
import sys
import numpy as np
from matplotlib import pyplot as plt
plt.ion()
sys.path.append('/Users/jsh3653/Documents/Austin/code/')
from ridge import ridge, utils
import logging
import random
import itertools as itools

# Show logging messages during STRF fitting about the correlations
# for each bootstrap of the training set, etc. (useful info)
logging.basicConfig(level=logging.DEBUG)

# For some reason when I set the logging above,
# I get a lot of annoying parso messages when I
# try to autocomplete unless I run the line below:
logging.getLogger('parso.python.diff').disabled = True

np.random.seed(0)
random.seed(0)

zs = lambda v: (v-v.mean(0))/v.std(0) ## z-score function

nmf_file='/Users/jsh3653/Documents/Austin/data/EEG/MT_TIMIT/all_out_ECoG_NMF_components.hf5'
data_dir='/Users/jsh3653/Documents/Austin/data/EEG/MT_TIMIT/'
subj = 'MT0006'   ## MT0017 is the worst - TIMIT. try also MT0016

raw=mne.io.read_raw_brainvision('/Users/jsh3653/Documents/Austin/data/EEG/MT_TIMIT/MT0008/downsampled_128/MT0008_DS128.vhdr')

stim_dict = dict()
resp_dict = dict()

with h5py.File(f'{data_dir}/fullEEGmatrix.hf5','r') as hf:
	print([k.split('.')[0] for k in hf['TIMIT'].keys()])
	names = [k.split('.')[0] for k in hf['TIMIT'].keys()]
	wav_names = [k for k in hf['TIMIT'].keys()]

	for name in names:
		# Responses are the EEG data for each sentence, averaged
		# across trials (dimension 0). Final dimensions are chans x ntimes
		# at 128 Hz sampling rate
		try:
			resp_dict[name] = hf[f'TIMIT/{name}.wav/resp/{subj}/epochs'][:].mean(0).T
			print(resp_dict[name].shape)
		except:
			print(f'No data for {name}')
		
with h5py.File(nmf_file, 'r') as hf:
	for name in resp_dict.keys():
		stim_dict[name] = scipy.signal.resample(hf[f'{name}/nmf'][:], resp_dict[name].shape[0])
		print(stim_dict[name].shape, resp_dict[name].shape)

for name in resp_dict.keys():
	resp_dict[name] = zs(resp_dict[name])
	stim_dict[name] = zs(stim_dict[name])

delays = [0]

test_set = ['fcaj0_si1479', 'fcaj0_si1804', 'fdfb0_si1948', 
			'fdxw0_si2141', 'fisb0_si2209', 'mbbr0_si2315', 
			'mdlc2_si2244', 'mdls0_si998', 'mjdh0_si1984', 
			'mjmm0_si625']

training_set = np.setdiff1d([k for k in resp_dict.keys()], test_set)

# print("Getting training and testing stim")
# tStim_temp = np.vstack(([stim_dict[r] for r in training_set]))
# vStim_temp = np.vstack(([stim_dict[r] for r in test_set]))

# print("Making delayed stim matrices")
# tStim = utils.make_delayed(tStim_temp, delays)
# vStim = utils.make_delayed(vStim_temp, delays)

# print("Getting training and testing resp")
# tResp = np.vstack(([resp_dict[r] for r in training_set]))
# vResp = np.vstack(([resp_dict[r] for r in test_set]))


chunklen = int(len(delays)*3) # We will randomize the data in chunks 
nchunks = np.floor(0.2*tStim.shape[0]/chunklen).astype('int')

# Regularization parameters (alphas - also sometimes called lambda)
alphas = np.hstack((0, np.logspace(2,8,20))) # Gives you 15 values between 10^2 and 10^8

nalphas = len(alphas)
use_corr = True # Use correlation between predicted and validation set as metric for goodness of fit
single_alpha = True # Use the same alpha value for all electrodes (helps with comparing across sensors)

nchunks = int(0.2*tResp.shape[0]/chunklen)  # 20% of training data

print("Fitting STRFs")
nboots=10
# wt, corrs, valphas, bcorrs, valinds, pred, pstim = ridge.bootstrap_ridge(tStim,
#                       tResp, vStim, vResp, alphas, nboots,
#                       chunklen, nchunks, single_alpha=True, corrmin=0.02)


###
#mne.viz.plot_topomap(corrs, raw.info)


## REVERSE MODEL

delays = [0]
print("Getting training and testing stim")
tStim_temp = np.vstack(([resp_dict[r] for r in training_set]))
vStim_temp = np.vstack(([resp_dict[r] for r in test_set]))

print("Making delayed stim matrices")
tStim = utils.make_delayed(tStim_temp, delays)
vStim = utils.make_delayed(vStim_temp, delays)

print("Getting training and testing resp")
tResp = np.vstack(([stim_dict[r] for r in training_set]))
vResp = np.vstack(([stim_dict[r] for r in test_set]))

print("Fitting STRFs")
nboots=10
wt, corrs, valphas, bcorrs, valinds, pred, pstim = ridge.bootstrap_ridge(tStim,
                      tResp, vStim, vResp, alphas, nboots,
                      chunklen, nchunks, single_alpha=True, corrmin=0.02)

clrs = [['r','m'], ['b', 'c']]
rtype = ['Onset', 'Sustained']

plt.figure(figsize=(1.0, 2.5))
for i in np.arange(2):
	plt.subplot(2,1,i+1)
	mne.viz.plot_topomap(wt[:,i], raw.info)
	plt.title(rtype[i])
plt.savefig(f'/Users/jsh3653/Documents/Austin/code/ECoG_to_EEG/{subj}_predicted_components_NMF_topos.pdf')


times = np.linspace(0, vResp.shape[0]/128., vResp.shape[0])
plt.figure(figsize=(4,2.5))
for i in np.arange(2):
	plt.subplot(2,1,i+1)
	plt.plot(times, vResp[:,i]/vResp[:,i].max(), color=clrs[i][0])
	plt.plot(times, pred[:,i]/pred[:,i].max(), color=clrs[i][1])
	plt.xlabel('Time (s)')
	plt.ylabel('Z-scored response')
	plt.title('%s: r=%2.2f'%(rtype[i], corrs[i]))

plt.tight_layout()
plt.savefig(f'/Users/jsh3653/Documents/Austin/code/ECoG_to_EEG/{subj}_predicted_components_NMF.pdf')
	
