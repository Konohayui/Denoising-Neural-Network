%% data
data = table2array(readtable("train_num.csv"));

%% train and val
rng(1024)

% split data into training and validation
% for modeling
% normalize training data by dividing 250
% to accelerate training time

idx = randperm(42000, 12600);
train = data(:, 2:end)/255;
train_labels = categorical(data(:, 1));

val = train(idx, 1:end);
val = reshape(val', 28, 28, 1, 12600);
val_labels = train_labels(idx, 1);

noise_mean = 0; noise_vars = [0, 0.5, 1, 1.5];
noise_var = noise_vars(4);

train(idx, :) = [];
train = imnoise(reshape(train', 28, 28, 1, 29400), "gaussian", noise_mean, noise_var);
train_labels(idx) = [];

%% network
network = [
    imageInputLayer([28 28 1], "Name", "Input")
    
    convolution2dLayer(3, 8, "Padding", "same", "Name", "Conv1")
    reluLayer("Name", "Relu1")
    maxPooling2dLayer(2, "Stride", 2, "Name", "Maxpool1")
    
    convolution2dLayer(3, 16, "Padding", "same", "Name", "Conv2")
    reluLayer("Name", "Relu2")
    maxPooling2dLayer(2, "Stride", 2, "Name", "Maxpool2")
    
    convolution2dLayer(3, 32, "Padding", "same", "Name", "Conv3")
    reluLayer("Name", "Relu3")
    maxPooling2dLayer(2, "Stride", 2, "Name", "Maxpool3")
    
    fullyConnectedLayer(10, "Name", "Dense")
    softmaxLayer("Name", "Output_Act")
    classificationLayer("Name", "Output")];


options = trainingOptions("adam",...
    "InitialLearnRate", 0.01,...
    "MaxEpochs", 4,...
    "Shuffle", "every-epoch",...
    "ValidationData", {val, val_labels},...
    "ValidationFrequency", 30,...
    "Verbose", false,...
    "ExecutionEnvironment", "gpu",...
    "Plots", "training-progress");

% analyzeNetwork(network)

%% training and prediction
model = trainNetwork(train, train_labels, network, options);

[Ypred_prob, Ypred_labels] = max(predict(model, val), [], 2);
Ypred_labels = Ypred_labels - 1;
correct_idx = find(val_labels == categorical(Ypred_labels));
incorrect_idx = find(val_labels ~= categorical(Ypred_labels));

%% validation results
figure
conmat = confusionchart(val_labels, categorical(Ypred_labels));

%% visualize convolution layer activation
conv_layers = ["Conv1", "Conv2", "Conv3"];

% image activation of a correct prediction
Ypred_prob(correct_idx(1))
val_labels(correct_idx(1))
Ypred_labels(correct_idx(1))
for layer = conv_layers
    fig = figure;
    conv_layer = activations(model, val(:,:,:,correct_idx(1)), layer);
    sz = size(conv_layer);
    conv_layer = reshape(conv_layer, [sz(1), sz(2), sz(3)]);
    conv_act = imtile(mat2gray(conv_layer));
    imshow(conv_act)
%     saveas(fig, layer+"noise_level="+noise_var+".png")
end

% image activation of an incorrect prediction
Ypred_prob(incorrect_idx(1))
val_labels(incorrect_idx(1))
Ypred_labels(incorrect_idx(1))
for layer = conv_layers
    fig = figure;
    conv_layer = activations(model, val(:,:,:,incorrect_idx(1)), layer);
    sz = size(conv_layer);
    conv_layer = reshape(conv_layer, [sz(1), sz(2), sz(3)]);
    conv_act = imtile(mat2gray(conv_layer));
    imshow(conv_act)
end

%% de-noise cnn model
DnCNNLayer = denoisingNetwork('DnCNN').Layers(2:end-1);

network2 = [
    imageInputLayer([28 28 1], "Name", "Input")
    
    DnCNNLayer

    fullyConnectedLayer(10, "Name", "Dense")
    softmaxLayer("Name", "Output_Act")
    classificationLayer("Name", "Output")];


options2 = trainingOptions("adam",...
    "InitialLearnRate", 0.01,...
    "MaxEpochs", 4,...
    "Shuffle", "every-epoch",...
    "ValidationData", {val, val_labels},...
    "ValidationFrequency", 30,...
    "Verbose", false,...
    "ExecutionEnvironment", "gpu",...
    "Plots", "training-progress");

% analyzeNetwork(network2)

%% Dn-CNN prediction
model2 = trainNetwork(train, train_labels, network2, options2);

[Ypred_prob2, Ypred_labels2] = max(predict(model2, val), [], 2);
Ypred_labels2 = Ypred_labels2 - 1;
correct_idx2 = find(val_labels == categorical(Ypred_labels2));
incorrect_idx2 = find(val_labels ~= categorical(Ypred_labels2));

%% visualize Dn-CNN image
% true label is 1
fig = figure;
conv_layer = activations(model2, train(:,:,:,1), "Conv20");
sz = size(conv_layer);
conv_layer = reshape(conv_layer, [sz(1), sz(2)]);
conv_act = imtile(mat2gray(conv_layer));
subplot(2, 3, 1)
imshow(reshape(data(1, 2:end), 28, 28, 1))
subplot(2, 3, 2)
imshow(train(:,:,:,1))
subplot(2, 3, 3)
imshow(conv_act)

% true label is 0
conv_layer = activations(model2, train(:,:,:,2), "Conv20");
sz = size(conv_layer);
conv_layer = reshape(conv_layer, [sz(1), sz(2)]);
conv_act = imtile(mat2gray(conv_layer));
subplot(2, 3, 4)
imshow(conv_act)
imshow(reshape(data(2, 2:end), 28, 28, 1))
subplot(2, 3, 5)
imshow(train(:,:,:,2))
subplot(2, 3, 6)
imshow(conv_act)
saveas(fig, "Dn-CNN-image-reconstruction.png")

