images = table2array(readtable("train_num.csv"));

%%
rng(1024)
num_sample = 20;
sample_index = randperm(1000, num_sample);

noise_mean = 0.0;
noise_var = [0, 0.5, 1, 1.5];

group = 0;
fig = figure;
for idx = [1:num_sample]
    temp_image = reshape(images(idx, 2:end), 28, 28);
    temp_noise_image = imnoise(temp_image, "gaussian", noise_mean, noise_var(4));
    subplot(5,8,idx+group)
    imshow(temp_image)
    subplot(5,8,idx+1+group)
    imshow(temp_noise_image)
    group = group + 1;
end

saveas(fig, "noise_level="+noise_var+".png")


