% CDM as per fornulas derived by me

clear all;
close all;
clc;
addpath('../libsvm-3.21/matlab');

%% Dataset preparation

dataload;
% xtest = 43*xtrain+51;
% ytest = ytrain;

rng(78);
indices = randperm(size(xtest,1));
sz_h = round(size(xtest,1)/2);
xtestL = xtest(indices(1:sz_h),:);
xtestU = xtest(indices(sz_h+1:end),:);
ytestL = ytest(indices(1:sz_h),:);
ytestU = ytest(indices(sz_h+1:end),:);

%% Parameters
c = 1;
c2 = 10000; % Need to be small since kernel argument is small
lam = 1;
lamr = 0;
eta = 0.001;

w = 1*ones(size(xtrain,2),1);
b = zeros(size(xtrain,2),1)+0;

%% Mean- variance correction
xtrain = mean_std(xtrain);
ytrain = mean_std(ytrain);
xtestL = mean_std(xtestL);
xtestU = mean_std(xtestU);
ytestL = mean_std(ytestL);
ytestU = mean_std(ytestU);

%%
p = 1;
X = [xtrain;xtestL];    
Y = [ytrain;ytestL];  
svm_model = svmtrain(Y,X,'-s 4');
[~,accuracy(:,p),~] = svmpredict(ytestU, xtestU, svm_model);
while p < 100
    p = p + 1;
    ynew = ytrain.*(xtrain*w) + xtrain*b;
    
    X = [xtrain;xtestL];
    Y = [ynew;ytestL];
    
%     X = mean_std(X);
%     Y = mean_std(Y);
    
    svm_model = svmtrain(Y,X,'-s 4');
    [~,Taccuracy(:,p),~] = svmpredict(Y, X, svm_model);
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
        w(k) = w(k) + eta*[-c2]*trace(Ltr2*Ltr*Ltr2*[kernel(ynew,ynew,c,c2).*(d_w)]);
        b(k) = b(k)+ eta*[-c2]*trace(Ltr2*Ltr*Ltr2*[kernel(ynew,ynew,c,c2).*(d_b)]);
        
        
        
        % Second expression
        for i = 1:size(ynew,1)
            for j = 1:size(ytestL,1)
                ep_w(i,j) = (ynew(i) - ytestL(j))*ytrain(i)*I(i,k);
                ep_b(i,j) = (ynew(i) - ytestL(j))*I(i,k);
            end
        end
        
        
        w(k) = w(k) + eta*[c2]*trace(Lte2*Ltetr*Ltr2*[kernel(ynew,ytestL,c,c2).*ep_w]);
        b(k) = b(k) + eta*[c2]*trace(Lte2*Ltetr*Ltr2*[kernel(ynew,ytestL,c,c2).*ep_b]);
    
    end
    if sum(isnan(w))
        break;
    end
        
    w = w + lamr*2*(w-1);
    b = b + lamr*2*b;
    
    storew(:,p) = w;
    storeb(:,p) = b;
end
plot(accuracy(2,1:end))
plot(Taccuracy(2,1:end))