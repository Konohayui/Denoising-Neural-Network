path = "C:\Users\Yu\Desktop\Thesis\MatLab Code\recordings\free-spoken-digit-dataset-master\recordings";
files = dir(path);

%% data
labels = zeros(3000, 1);
sounds = zeros(3000, 18262);
fs = 8000;

for idx = 3:3002
    file_name = files(idx).name;
    labels(idx-2) = str2double(file_name([1]));
    [y, fs] = audioread(path + "\" + file_name);
    sounds(idx-2,1:length(y)) = y;
end

%% sample
[y, fs] = audioread(path + "\" + files(5).name);
% sound(y)
% plot(y)

%% noise sample
noises = 2*sqrt(0.005)*rand(size(y)) - sqrt(0.005);
y2 = y + noises;
% sound(y2)
% plot(y2)

%% wavelet decomposition on sample signal
signal = zeros(1, 200);
signal(1:50) = 1;
signal(101:150) = 1;

wavelet = "db1";
[c,l] = wavedec(signal,4,wavelet);
approx = appcoef(c,l,wavelet);
[cd1,cd2,cd3,cd4] = detcoef(c,l,[1 2 3 4]);
fig = figure;
subplot(6,1,1)
plot(approx)
title('Approximation Coefficients')
subplot(6,1,2)
plot(cd4)
title('Level 4 Detail Coefficients')
subplot(6,1,3)
plot(cd3)
title('Level 3 Detail Coefficients')
subplot(6,1,4)
plot(cd2)
title('Level 2 Detail Coefficients')
subplot(6,1,5)
plot(cd1)
title('Level 1 Detail Coefficients')
subplot(6,1,6)
plot(signal)
title('Sample signal')

%% wavelet docomposition audio
wavelet = "db3";
[c,l] = wavedec(y,4,wavelet);
approx = appcoef(c,l,wavelet);
[cd1,cd2,cd3,cd4] = detcoef(c,l,[1 2 3 4]);
fig = figure;
subplot(6,1,1)
plot(approx)
title('Approximation Coefficients')
subplot(6,1,2)
plot(cd4)
title('Level 4 Detail Coefficients')
subplot(6,1,3)
plot(cd3)
title('Level 3 Detail Coefficients')
subplot(6,1,4)
plot(cd2)
title('Level 2 Detail Coefficients')
subplot(6,1,5)
plot(cd1)
title('Level 1 Detail Coefficients')
subplot(6,1,6)
plot(y)
title('Original Audio')

%% wavelet docomposition noise audio
[c,l] = wavedec(y2,4,wavelet);
approx = appcoef(c,l,wavelet);
[cd1,cd2,cd3,cd4] = detcoef(c,l,[1 2 3 4]);
fig = figure;
subplot(6,1,1)
plot(approx)
title('Approximation Coefficients')
subplot(6,1,2)
plot(cd4)
title('Level 4 Detail Coefficients')
subplot(6,1,3)
plot(cd3)
title('Level 3 Detail Coefficients')
subplot(6,1,4)
plot(cd2)
title('Level 2 Detail Coefficients')
subplot(6,1,5)
plot(cd1)
title('Level 1 Detail Coefficients')
subplot(6,1,6)
plot(y2)
title('Original Noise Audio')

%% save reconstruct audio data to csv format
writematrix(sounds, "sounds.csv")
writematrix(labels, "labels.csv")

%%
path = "C:\Users\Yu\Desktop\Thesis\MatLab Code\recordings\free-spoken-digit-dataset-master\recordings\";
file_name = "1_george_1.wav";

[y, Fs] = audioread(path+file_name);

src = dsp.SampleRateConverter("InputSampleRate", Fs,...
                              "OutputSampleRate", Fs,...
                              "Bandwidth", 7920);
y = src(y);
reset(src);
                          
var = sqrt(0.005);
noises = 2*var*rand(size(y)) - var;

noise_y = y + noises;

windowLength = 256;
win = hamming(windowLength,"periodic");
overlap = round(0.75 * windowLength);
ffTLength = windowLength;
fs = 8000;
numFeatures = ffTLength/2 + 1;
numSegments = 8;


%%
t = (1/fs)*(0:numel(y)-1);

fig = figure;
subplot(3,2,1), plot(t, y)
title("Clean Audio")
subplot(3,2,2), plot(t, noise_y)
title("Noisy Audio")
subplot(3,2,3), spectrogram(y, win, overlap, ffTLength, fs)
title("Clean Audio Spectrogram")
subplot(3,2,4), spectrogram(noise_y, win, overlap, ffTLength, fs)
title("Noisy Audio Spectrogram")
subplot(3,2,5), stft(y, fs, 'Window',win,'OverlapLength',overlap,'FFTLength',ffTLength)
title("Clean STFT Audio Spectrogram")
subplot(3,2,6), stft(noise_y, fs, 'Window',win,'OverlapLength',overlap,'FFTLength',ffTLength)
title("Noisy STFT Audio Spectrogram")

saveas(fig, "stft_audio.png")


%%
fig = figure;
subplot(3,1,1), plot(t, y)
title("Clean Audio")
subplot(3,1,2), stft(y, fs, 'Window',win,'OverlapLength',overlap,'FFTLength',ffTLength)
title("Clean Audio STFT with 256 Window Length")
windowLength_ = windowLength/4;
win_ = hamming(windowLength_,"periodic");
overlap_ = round(0.75 * windowLength_);
ffTLength_ = windowLength_;
subplot(3,1,3), stft(y, fs, 'Window',win_,'OverlapLength',overlap_,'FFTLength',ffTLength_)
title("Clean Audio STFT with 64 Window Length")

saveas(fig, "stft_audio_wind.png")


%%
cleanSTFT = stft(y, fs, 'Window',win,'OverlapLength',overlap,'FFTLength',ffTLength);
noisySTFT = stft(noise_y, fs, 'Window',win,'OverlapLength',overlap,'FFTLength',ffTLength);
cleanSTFT = abs(cleanSTFT(numFeatures-1:end,:));
noisySTFT = abs(noisySTFT(numFeatures-1:end,:));

noisySTFT = [noisySTFT(:,1:numSegments - 1), noisySTFT];
stftSegments = zeros(numFeatures, numSegments , size(noisySTFT,2) - numSegments + 1);
for index = 1:size(noisySTFT,2) - numSegments + 1
    stftSegments(:,:,index) = (noisySTFT(:,index:index + numSegments - 1)); 
end

fig = figure;
imagesc(stftSegments(:,:,1))
title("Noisy STFT Audio Spetrogram Segment One")
saveas(fig, "stft_audio_segment.png")

size(cleanSTFT)
size(stftSegments)

