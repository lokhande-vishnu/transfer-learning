% CDM by modifying xtrain

clear all;
close all;
clc;
addpath('../libsvm-3.21/matlab');

%% Dataset preparation

dataload;
xtest = sin(3*xtrain+4);
ytest = ytrain;

rng(78);
indices = randperm(size(xtest,1));
sz_h = round(size(xtest,1)/2);
xtestL = xtest(indices(1:sz_h),:);
xtestU = xtest(indices(sz_h+1:end),:);
ytestL = ytest(indices(1:sz_h),:);
ytestU = ytest(indices(sz_h+1:end),:);

%% Parameters
c = 1;
c2 = 10000; % sigma squared
lam = 1;
lamr = 0;
eta = 1;

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
[~,Taccuracy(:,p),~] = svmpredict(Y, X, svm_model);
[~,accuracy(:,p),~] = svmpredict(ytestU, xtestU, svm_model);
while p < 10000
    p = p + 1;
    ynew = ytrain.*(xtrain*w) + xtrain*b;
    xnew = 
    
    X = [xtrain;xtestL];
    Y = [ynew;ytestL];
    
    svm_model = svmtrain(Y,X,'-s 4');
    [~,Taccuracy(:,p),~] = svmpredict(Y, X, svm_model);
    [~,accuracy(:,p),~] = svmpredict(ytestU, xtestU, svm_model);
    
    %%%% Updating w and b
    Ltr = kernel(xtrain,xtrain,c,c2);
    Ltr2 = inv(Ltr+lam*eye(size(xtrain,1)));
    Lte = kernel(xtestL,xtestL,c,c2);
    Ltetr = kernel(xtestL,xtrain,c,c2);
    Lte2 = inv(Lte+lam*eye(size(xtestL,1)));
    
    L(:,p) = trace(Ltr2*kernel(ynew,ynew,c,c2)*Ltr2*Ltr) - 2*trace(Ltr2*kernel(ynew,ytestL,c,c2)*Lte2*kernel(xtestL,xtrain,c,c2));
    for k = 1:size(w,1)        
        % first expression
        for i = 1:size(ynew,1)
            for j = 1:size(ynew,1)
                d_w(i,j) = (-1/c2)*(ynew(i) - ynew(j))*(ytrain(i)*I(i,k) - ytrain(j)*I(j,k));
                d_b(i,j) = (-1/c2)*(ynew(i) - ynew(j))*(I(i,k) - I(j,k));
            end
        end        
        dl_dk = Ltr2*Ltr'*Ltr2;
        w(k) = w(k) + eta*trace(dl_dk'*[kernel(ynew,ynew,c,c2).*(d_w)]);
        b(k) = b(k)+ eta*trace(dl_dk'*[kernel(ynew,ynew,c,c2).*(d_b)]);
                
        % Second expression
        for i = 1:size(ynew,1)
            for j = 1:size(ytestL,1)
                ep_w(i,j) = (-1/c2)*(ynew(i) - ytestL(j))*ytrain(i)*I(i,k);
                ep_b(i,j) = (-1/c2)*(ynew(i) - ytestL(j))*I(i,k);
            end
        end
        dl_dk2 = 2*Ltr2*Ltetr'*Lte2;        
        w(k) = w(k) + eta*trace(dl_dk2'*[kernel(ynew,ytestL,c,c2).*ep_w]);
        b(k) = b(k) + eta*trace(dl_dk2'*[kernel(ynew,ytestL,c,c2).*ep_b]);
    end
    
    
    if sum(isnan(w))
        break;
    end
        
    w = w + lamr*2*(w-1);
    b = b + lamr*2*b;
    
    storew(:,p) = w;
    storeb(:,p) = b;
    
end
figure(1);
subplot(3,1,1);
plot(L(1,2:end),'-*');
subplot(3,1,2);
plot(Taccuracy(2,1:end),'-*');
subplot(3,1,3);
plot(accuracy(2,1:end),'-*');