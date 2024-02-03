clear all;
close all;
clc;

lbl = loadMNISTLabels('MNIST_dataset\train-labels.idx1-ubyte');
imgs = loadMNISTImages('MNIST_dataset\train-images.idx3-ubyte');

rows = 28;
cols = rows;
num_imgs = 60000;

%################ 1 ##################    
train_percentage = 0.45;

train_set_lbls = (lbl(1:num_imgs*train_percentage));  % Convert to categorical
test_set_lbls = (lbl((num_imgs*train_percentage+1):num_imgs));

train_set_imgs = imgs(:,1:num_imgs*train_percentage); % Adjust the dimensions
test_set_imgs = imgs(:,(num_imgs*train_percentage+1):num_imgs); % Adjust the dimensions

inputSize = rows * cols;  % 28x28 = 784
hiddenLayerSize = 45;     % Number of neurons in the hidden layer 
outputSize = 10;          % Number of output classes (0-9)

X_train = reshape(train_set_imgs, [rows, cols, 1, size(train_set_imgs, 2)]);
X_test = reshape(test_set_imgs, [rows, cols, 1, size(test_set_imgs, 2)]);

% Convert labels to categorical
Y_train = categorical(train_set_lbls);
Y_test = categorical(test_set_lbls);

% Define the layers of the neural network
layers = [
    imageInputLayer([rows, cols, 1])
    fullyConnectedLayer(hiddenLayerSize)
    reluLayer
    fullyConnectedLayer(outputSize)
    softmaxLayer
    classificationLayer
];

% Set the training options
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {X_test, Y_test}, ...
    'Plots', 'training-progress');

% Train the neural network OR reload already trained nn
%net = trainNetwork(X_train, Y_train, layers, options);

savedNetworkFile = '45hidden9xtrain.mat';       % 9*0.05 = 0.45
%save(savedNetworkFile, 'net');
net_load = load(savedNetworkFile);

% Make predictions on the test set
Y_pred = classify(net_load.net, X_test);

% Evaluate the performance
accuracy = sum(Y_pred == Y_test) / numel(Y_test);
fprintf('Accuracy on the test set: %.2f%%\n', accuracy * 100);

trueLabels = double(Y_test) - 1;  % Subtract 1 to convert categorical labels 1-10 to 0-9
predictedLabels = double(Y_pred) - 1;
confMat = confusionmat(trueLabels, predictedLabels)