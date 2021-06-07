path = "C:\Users\Yu\Desktop\Thesis\MatLab Code\recordings\free-spoken-digit-dataset-master\recordings";
files = dir(path);

%% data
labels = zeros(3000, 1);
sounds = zeros(3000, 18262);

for idx = 3:3002
    file_name = files(idx).name;
    labels(idx-2) = str2double(file_name([1]));
    [y, fs] = audioread(path + "\" + file_name);
    sounds(idx-2,1:length(y)) = y;
end

%% sample
[y, Fs] = audioread(path + "\" + files(5).name);
sound(y)
plot(y)

%% noise sample
y2 = imnoise(y, "gaussian", 0, 0.01);
sound(y2)
plot(y2)

%%
writematrix(sounds, "sounds.csv")
writematrix(labels, "labels.csv")

