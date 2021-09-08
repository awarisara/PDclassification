import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import os
import scipy
from scipy import signal
import warnings

file = []
Energy = []
max_db = 60

#read audio files (/aa/ phenome for 10 seconds)
path = 'audio_files'
warnings.filterwarnings(action = "ignore")
x, sr = librosa.load(path,sr=None,duration=10.000)
warnings.filterwarnings(action = "default")

# audio length should be more 9 seconds.
if len(x) < 9*sr:
	continue
else:
	for ii in range(len(x)//(sr//2)):
		#separated to the short segments. we used 10 seconds and sr = 44100 so we got 20 segments.
        xsqr = x[ii*sr//2:(ii+1)*sr//2]**2
        Energy.append(xsqr.sum())
        Energy = np.array(Energy)
		
		#Finding normal segments
		#defined one standard deviation.
        datamean = Energy.mean()
        datastd = Energy.std()
		maxeng = datamean + datastd
        mineng = datamean - datastd
        
        for jj in range(len(Energy)-1): #last window(20th) size is not equal other windows so we uses 19 windows.
		
		#defined the creiteria.
            if (Energy[jj] >= mineng) and (Energy[jj+1] >= mineng) and (Energy[jj] <= maxeng) and (Energy[jj+1] <= maxeng):
                #save normal segment
				x_new = x[jj*sr//2:jj*sr//2+sr]
				print(x_new.shape)
				
				#STFT caluculation with scipy library
                f, t, Zxx = signal.stft(x_new, sr, boundary = None, nperseg= sr//20)
                Zxx = np.abs(Zxx)
                #print(Zxx.shape)
				
				#set parameters to generate spectrogram.
                min_value = 10**(-max_db/20)*Zxx.max()
                Zxx = np.where(Zxx < min_value,min_value,Zxx)

				#visualized in log-scale, dB unit.
                Zxxdb = 20*np.log10(np.abs(Zxx))
                datamin=Zxxdb.min()
                datamax=Zxxdb.max()
                data = Zxxdb-datamax
                data = np.where(data<-max_db, -max_db,data)
                data = np.uint8((data+max_db)*255//max_db)
				
				#after we visualized spectrogram, the informative signal is at 0-6000Hz so we set the cut-off frequency at 6000 Hz.
				out = data[0:6000//20,:]
                out = np.array(out)
				
				#save all 1-sec segments in one array.
                file.append(out)

#save file into .npy				
path = 'destination'
filename = 'your filename'
np.save(path+filename,file)
