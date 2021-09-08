# Parkinson's Disease classification

An audio signal was converted to a spectrogram by calculation Short-Time Fourier Transform in order to classify participants with and without Parkinson's disease through convolutional neural network in deep learning.

# Methodology
## Preprosessing data
- Audio waveforms are separated to short segments and calculate energy of each segment by using equation: ![energy equaition](https://latex.codecogs.com/png.latex?E_k%20%3D%20%5Cmathbf%20x_k%20%5Ccdot%20%5Cmathbf%20x_k%5ET)
- Ek is used to determine the normality of segment of the audio. The Ek within one standard deviation of the mean is considered normal, the data is 1-second normal segment which is two consecutive normal segments.
- After we selected the normal segments, each segment was converted to spectrogram by using STFT calculation since a spectrogram is magnitude of STFT and it visualizes in log-scale (decibels). ![dB_equaition](https://latex.codecogs.com/png.latex?Amplitude%20%28dB%29%20%3D%2010%20%5Ctimes%20log%7B%28%7CSTFT%7C%29%7D%5E2)
- Next, the log-scale spectrogram is normalized to 0 to 255 like an image.
## Training dataset
- Finally, the 1 second spectrogram takes into CNN models (LeNet-5, VGGNet-16, and ResNet-50)

# Dataset Availability Statement
Our dataset contributed by users of the Parkinson mPower mobile application as part of the mPower study developed by Sage Bionetworks and described in Synapse (doi:10.7303/syn4993293). Since the information used in this research is not accessible to all researchers. Please contact the corresponding author for more information. 
