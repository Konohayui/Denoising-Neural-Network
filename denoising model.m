%% load data
rng(1024)
audio = table2array(readtable("sounds.csv"));
% labels = table2array(readtable("labels.csv"));

%% wavelet denoising
fig = figure;
sample = audio(1,1:2500);
subplot(3, 1, 1)
plot(sample)
title("clean audio")

var = sqrt(0.005);
noise_sample = sample + 2*var*rand(size(sample)) - var;
subplot(3, 1, 2)
plot(noise_sample)
title("noise audio")

denoise_sample = wdenoise(noise_sample);
subplot(3, 1, 3)
plot(denoise_sample)
title("wavelet transform denoise audio")

saveas(fig, "wavelet_denoise_result.png")

%% compute error
err = immse(denoise_sample, sample);
err

%% load data
% original code: https://www.mathworks.com/help/deeplearning/ug/denoise-speech-using-deep-learning-networks.html
% num_data = size(audio);
% rand_ind = randperm(num_data(1), num_data(1));
% audio = audio(rand_ind, :);
% audio = audio(1:100, :);

downloadFolder = "C:\Users\Yu\Desktop\Thesis\MatLab Code\recordings\";
dataFolder = fullfile(downloadFolder,'free-spoken-digit-dataset-master');
ads = audioDatastore(fullfile(dataFolder,'recordings'),'IncludeSubfolders',true);
ads = shuffle(ads);
ads = subset(ads,1:1000);

var = sqrt(0.005);
noise = 2*var*rand([18262, 1]) - var;

src = dsp.SampleRateConverter("InputSampleRate", fs,...
                              "OutputSampleRate", fs,...
                              "Bandwidth", 7920);
                          
%%
T = tall(ads);

var = sqrt(0.005);
noise = 2*var*rand([18262, 1]) - var;
                          
windowLength = 256;
win = hamming(windowLength,"periodic");
overlap = round(0.75 * windowLength);
ffTLength = windowLength;
fs = 8000;
numFeatures = ffTLength/2 + 1;
numSegments = 8;
                          
% targets = []; predictors = [];
% 
% for idx = 1:100
%     [clean_audio, info] = read(ads);
%     [temp1, temp2] = HelperGenerateSpeechDenoisingFeatures(clean_audio,noise,src);
%     targets = [targets, temp1];
%     predictors = [predictors, temp2];
% end

[targets,predictors] = cellfun(@(x)HelperGenerateSpeechDenoisingFeatures(x,noise,src), T, "UniformOutput",false);
[targets,predictors] = gather(targets,predictors);

%% normalized and reshape data
predictors    = cat(3, predictors{:});
targets       = cat(2, targets{:});
noisyMean     = mean(predictors(:));
noisyStd      = std(predictors(:));
predictors(:) = (predictors(:)-noisyMean)/noisyStd;
cleanMean     = mean(targets(:));
cleanStd      = std(targets(:));
targets(:)    = (targets(:)-cleanMean)/cleanStd;

predictors  = reshape(predictors,size(predictors,1),size(predictors,2),1,size(predictors,3));
targets     = reshape(targets,1,1,size(targets,1),size(targets,2));

%% split data
inds                = randperm(size(predictors,4));
L                   = round(0.99 * size(predictors,4));
trainPredictors     = predictors(:,:,:,inds(1:L));
trainTargets        = targets(:,:,:,inds(1:L));
validatePredictors  = predictors(:,:,:,inds(L+1:end));
validateTargets     = targets(:,:,:,inds(L+1:end));


%% model 
layers = [imageInputLayer([numFeatures,numSegments])
          convolution2dLayer([9 8],18,"Stride",[1 100],"Padding","same")
          batchNormalizationLayer
          reluLayer
          
          repmat(...
          [convolution2dLayer([5 1],30,"Stride",[1 100],"Padding","same")
          batchNormalizationLayer
          reluLayer
          
          convolution2dLayer([9 1],8,"Stride",[1 100],"Padding","same")
          batchNormalizationLayer
          reluLayer
          
          convolution2dLayer([9 1],18,"Stride",[1 100],"Padding","same")
          batchNormalizationLayer
          reluLayer], 4,1) % repeat this sequence 4 times
          
          convolution2dLayer([5 1],30,"Stride",[1 100],"Padding","same")
          batchNormalizationLayer
          reluLayer
          
          convolution2dLayer([9 1],8,"Stride",[1 100],"Padding","same")
          batchNormalizationLayer
          reluLayer
          
          convolution2dLayer([129 1],1,"Stride",[1 100],"Padding","same")
          
          regressionLayer
          ];

%% parameter
miniBatchSize = 32;
options = trainingOptions("adam", ...
    "MaxEpochs",3, ...
    "InitialLearnRate",1e-5,...
    "MiniBatchSize",miniBatchSize, ...
    "Shuffle","every-epoch", ...
    "Plots","training-progress", ...
    "Verbose",false, ...
    "ValidationPatience",Inf,...
    "ValidationFrequency",floor(size(trainPredictors,4)/miniBatchSize),...
    "LearnRateSchedule","piecewise",...
    "LearnRateDropFactor",0.9,...
    "LearnRateDropPeriod",1,...
    "ValidationData",{validatePredictors,permute(validateTargets,[3 1 2 4])});miniBatchSize = 32;

%% training
denoiseNetFullyConvolutional = trainNetwork(trainPredictors,permute(trainTargets,[3 1 2 4]),layers,options);

%%
save("denoisenet.mat", "denoiseNetFullyConvolutional")

%% prediction
D = 1;

reset(ads);
[denoisedAudio, cleanAudio, noisyAudio] = testNetwork(ads, noise, src, D); 

fig = figure;
t = (1/fs) * ( 0:numel(denoisedAudio)-1);

subplot(321)
plot(t,cleanAudio(1:numel(denoisedAudio)))
title("Clean Speech")
grid on
subplot(322)
spectrogram(cleanAudio, win, overlap, ffTLength, fs);
title("Clean Speech")
grid on
subplot(323)
plot(t,noisyAudio(1:numel(denoisedAudio)))
title("Noisy Speech")
grid on
subplot(324)
spectrogram(noisyAudio, win, overlap, ffTLength,fs);
title("Noisy Speech")
grid on
subplot(325)
plot(t,denoisedAudio)
title("Denoised Speech")
grid on
xlabel("Time (s)")
subplot(326)
spectrogram(denoisedAudio, win, overlap, ffTLength,fs);
title("Denoised Speech")
grid on

saveas(fig, "denoisemodel_results.png")


%%
function [targets,predictors] = HelperGenerateSpeechDenoisingFeatures(audio,noise, src)
% HelperGenerateSpeechDenoisingFeatures: Get target and predictor STFT
% signals for speech denoising.
% audio: Input audio signal
% noise: Input noise signal
% src:   Sample rate converter
% Copyright 2018 The MathWorks, Inc.
WindowLength = 256;
win          = hamming(WindowLength,'periodic');
Overlap      = round(0.75 * WindowLength);
FFTLength    = WindowLength;
Fs           = 8e3;
NumFeatures  = FFTLength/2 + 1;
NumSegments  = 8;

D            = 1; % Decimation factor % modify to 1 from 48/8
L            = floor( numel(audio)/D);
audio        = audio(1:D*L);
audio   = src(audio);
reset(src);

randind      = randi(abs(numel(noise) - numel(audio)) , [1 1]);
noiseSegment = noise(randind : randind + numel(audio) - 1);
noisePower   = sum(noiseSegment.^2);
cleanPower   = sum(audio.^2);
noiseSegment = noiseSegment .* sqrt(cleanPower/noisePower);
noisyAudio   = audio + noiseSegment;

cleanSTFT = stft(audio,'Window',win,'OverlapLength',overlap,'FFTLength',ffTLength);
cleanSTFT = abs(cleanSTFT(numFeatures-1:end,:));
noisySTFT = stft(noisyAudio,'Window',win,'OverlapLength',overlap,'FFTLength',ffTLength);
noisySTFT = abs(noisySTFT(numFeatures-1:end,:));
noisySTFTAugmented    = [noisySTFT(:,1:NumSegments-1) noisySTFT];
 
STFTSegments = zeros( NumFeatures, NumSegments , size(noisySTFTAugmented,2) - NumSegments + 1);
for index     = 1 : size(noisySTFTAugmented,2) - NumSegments + 1
    STFTSegments(:,:,index) = noisySTFTAugmented(:,index:index+NumSegments-1);
end
targets    = cleanSTFT;
predictors = STFTSegments;

end

function [denoisedAudio, cleanAudio, noisyAudio] = testNetwork(ads, noise, src, D)

WindowLength = 256;
win          = hamming(WindowLength,"periodic");
Overlap      = round(0.75 * WindowLength);
FFTLength    = WindowLength;
inputFs      = 8e3;
Fs           = 8e3;
NumFeatures  = FFTLength/2 + 1;
NumSegments  = 8;
%%
s = load("denoisenet.mat");
denoiseNetFullyConvolutional = s.denoiseNetFullyConvolutional;
cleanMean = 0.1716;
cleanStd  = 0.6039;
noisyMean = 0.5557;
noisyStd  = 0.6907;

%% Shuffle the files in the datastore
ads = shuffle(ads);
% Read the contents of a file from the datastore
[cleanAudio,info] = read(ads);
% Make sure the audio length is a multiple of the sample rate converter decimation factor
L            = floor( numel(cleanAudio)/D);
cleanAudio   = cleanAudio(1:D*L);
% Convert the audio signal to 8 kHz:
cleanAudio   = src(cleanAudio);
reset(src)
% In this testing stage, you corrupt speech with washing machine noise not used in the training stage.

%% Create a random noise segment from the washing machine noise vector.
randind      = randi(numel(noise) - numel(cleanAudio) , [1 1]);
noiseSegment = noise(randind : randind + numel(cleanAudio) - 1);
% Add noise to the speech signal such that the SNR is 0 dB.
noisePower   = sum(noiseSegment.^2);
cleanPower   = sum(cleanAudio.^2);
noiseSegment = noiseSegment .* sqrt(cleanPower/noisePower);
noisyAudio   = cleanAudio + noiseSegment;

%% Use spectrogram to generate magnitude STFT vectors from the noisy audio signals:
noisySTFT  = spectrogram(noisyAudio, win, Overlap, FFTLength,Fs);
noisyPhase = angle(noisySTFT);
noisySTFT  = abs(noisySTFT);

%% Generate the 8-segment training predictor signals from the noisy STFT. The overlap between consecutive predictors is 7 segments.
noisySTFT    = [noisySTFT(:,1:NumSegments-1) noisySTFT];
predictors = zeros( NumFeatures, NumSegments , size(noisySTFT,2) - NumSegments + 1);
for index     = 1 : size(noisySTFT,2) - NumSegments + 1
    predictors(:,:,index) = noisySTFT(:,index:index+NumSegments-1);
end

%% Normalize the predictors by the mean and standard deviation computed in the training stage:
predictors(:) = (predictors(:) - noisyMean) / noisyStd;
% Compute the denoised magnitude STFT by using predict with the two trained networks.
predictors = reshape(predictors,[NumFeatures, NumSegments,1,size(predictors,3)]);
STFTFullyConvolutional = predict(s.denoiseNetFullyConvolutional, predictors);
% Scale the outputs by the mean and standard deviation used in the training stage.
STFTFullyConvolutional(:) = cleanStd * STFTFullyConvolutional(:) +  cleanMean;

%% Compute the denoised speech signals. istft performs the inverse STFT. Use the phase of the noisy STFT vectors to reconstruct the time-domain signal.
denoisedAudio = istft(squeeze(STFTFullyConvolutional).* exp(1j*noisyPhase), ...
    "Overlap", WindowLength-Overlap, "FFTLength", FFTLength, "Window", win);
% denoisedAudio = istft(squeeze(STFTFullyConvolutional).* exp(1j*noisyPhase));
end
