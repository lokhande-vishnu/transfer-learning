clear all;
close all;
clc;
addpath('../libsvm-3.21/matlab');

%% Dataset preparation

load fisheriris.mat
xtrain = meas(1:100,:);
ytrain = [1*ones(50,1);2*ones(50,1)];%;3*ones(50,1)];

rng(78);
xtest = 3*xtrain + 4;
indices = randperm(100);
xtestL = xtest(indices(1:50),:);
xtestU = xtest(indices(51:end),:);
ytestL = ytrain(indices(1:50),:);
ytestU = ytrain(indices(51:end),:);

%% Parameters
c = 1;
c2 = 0.0005;
lam = 1;
lamr = 1;
eta = 1;

w = ones(size(xtrain,2),1);
b = zeros(size(xtrain,2),1);

%% Mean- variance correction
xtrain = mean_std(xtrain);
xtestL =  mean_std(xtestL);
xtestU =  mean_std(xtestU);

%%
p = 0;
while p < 10
    p = p + 1;
    ynew = ytrain.*(xtrain*w) + xtrain*b;
    
    X = [xtrain;xtestL];
    Y = [ynew;ytestL];
    
    svm_model = svmtrain(Y,X);
    [~,accuracy(:,p),~] = svmpredict(ytestU, xtestU, svm_model);
    
    %%%% Updating w and b
    Ltr = kernel(xtrain,xtrain,c,c2);
    Ltr2 = inv(Ltr+lam*eye(size(ynew,1)));
    Lte = kernel(xtestL,xtestL,c,c2);
    Ltetr = kernel(xtestL,xtrain,c,c2);
    Lte2 = inv(Lte+lam*eye(size(ytestL,1)));
    for k = 1:size(w,1)
        
        % first expression
        for i = 1:size(ynew,1)
            eps(i) = ytrain(i,:)*[xtrain(i,:)*w] + xtrain(i,:)*b - ytrain(i)*xtrain(i,k)*w(k,1) - xtrain(i,k)*b(k,1);
            for j = 1:size(ynew,1)
                eps(j) = ytrain(j,:)*xtrain(j,:)*w + xtrain(j,:)*b - ytrain(j)*xtrain(j,k)*w(k,1) - xtrain(i,k)*b(k,1);
                d_w_ynewynewt(i,j) = 2*w(k)*xtrain(i,k)*xtrain(j,k)*ytrain(i)*ytrain(j) + b(k)*[xtrain(i,k)*xtrain(j,k)*ytrain(j) + xtrain(i,k)*ytrain(i)*xtrain(j,k)] + eps(i)*xtrain(j,k)*ytrain(j) + xtrain(i,k)*ytrain(i)*eps(j);
                d_w(i,j) = (ynew(i) - ynew(j))*(ytrain(i)*I(i,k) - ytrain(j)*I(j,k));
                d_b_ynewynewt(i,j) = 2*b(k)*xtrain(i,k)*xtrain(j,k) + w(k)*[xtrain(i,k)*xtrain(j,k)*ytrain(j) + xtrain(i,k)*ytrain(i)*xtrain(j,k)] + eps(i)*xtrain(j,k) + xtrain(i,k)*eps(j);
                d_b(i,j) = (ynew(i) - ynew(j))*(I(i,k) - I(j,k));
                
            end
        end
        
        %w(k) = w(k)+ [-c2]*trace(Ltr2*Ltr*Ltr2*kernel(ynew,ynew,c,c2)*(d_w_ynewynewt));
        %b(k) = b(k)+ [-c2]*trace(Ltr2*Ltr*Ltr2*kernel(ynew,ynew,c,c2)*(d_b_ynewynewt));
        w(k) = w(k) + [-c2]*trace(Ltr2*Ltr*Ltr2*[kernel(ynew,ynew,c,c2).*(d_w)]);
        b(k) = b(k)+ [-c2]*trace(Ltr2*Ltr*Ltr2*[kernel(ynew,ynew,c,c2).*(d_b)]);
        
        
        
        % Second expression
        for i = 1:size(ynew,1)
            for j = 1:size(ytestL,1)
                ep_w(i,j) = (ynew(i) - ytestL(j))*ytrain(i)*I(i,k);
                ep_b(i,j) = (ynew(i) - ytestL(j))*I(i,k);
            end
        end
        
        
        w(k) = w(k) + [2*c2]*trace(Lte2*Ltetr*Ltr2*[kernel(ynew,ytestL,c,c2).*ep_w]);
        b(k) = b(k) + [2*c2]*trace(Lte2*Ltetr*Ltr2*[kernel(ynew,ytestL,c,c2).*ep_b]);
        
    end
    w = w + lamr*2*(w-1);
    b = b + lamr*2*b;
    
    store(:,p) = w;
    store2(:,p) = b;
end
