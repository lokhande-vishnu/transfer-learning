%% Dataset loading

% %% fisheriris dataset
% load fisheriris.mat
% xtrain = meas(1:100,:);
% ytrain = [1*ones(50,1);2*ones(50,1)];%;3*ones(50,1)];

% %% machine data
% http://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/
% load('machinedata.mat');
% xtrain = machinedata(:,1:end-1);
% ytrain = machinedata(:,end);

% housing
% https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
load('../housing.mat');
len = round(size(housing,1)/2);
xtrain = housing(1:len,1:end-1);
xtrain = mean_std(xtrain);
w1 = 0.13*ones(size(xtrain,2),1);
ytrain = xtrain*w1;
xtest = housing(len+1:end,1:end-1);
xtest = mean_std(xtest);
ytest = (xtest + xtrain)*w1;

% % parameters
% c = 0.001;
% c2 = 0.000005; % Need to be small since kernel argument is small
% lam = 100;
% lamr = 1;
% eta = 1;

% %% Paper dataset
% xtrain = [-3:0.2:-1 -0.5:0.5:0 3:0.2:5]';
% ytrain = sin(xtrain);
% xtest = [-5:0.35:5]';
% ytest = sin(xtest+1);

% Paper dataset 2
% Parameters
% c = 100;
% c2 = 0.00005; % Need to be small since kernel argument is small
% lam = 1;
% lamr = 0;
% eta = 0.001;

%% Simple dataset
% xtrain = [-3:0.2:-1 -0.5:0.5:0 3:0.2:6]';
% w1 = 13;
% ytrain = xtrain*w1;
% xtest = [-5:0.35:5]';
% ytest = (xtest + xtrain)*w1;
