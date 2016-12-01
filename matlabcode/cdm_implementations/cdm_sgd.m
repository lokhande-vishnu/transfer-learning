% CDM as per the formulas by the book
% stochastic gradient descent


clear all;
close all;
clc;
addpath('../libsvm-3.21/matlab');

%% Dataset preparation

dataload;
% wk = 357.23*ones(size(xtrain,2),1);
% bk = zeros(size(xtrain,2),1)+123.45;
% xtest = xtrain;
% ytest = 4*ytrain+5;

rng(78);
indices = randperm(size(xtest,1));
sz_h = round(size(xtest,1)/2);
xtestL = xtest(indices(1:sz_h),:);
xtestU = xtest(indices(sz_h+1:end),:);
ytestL = ytest(indices(1:sz_h),:);
ytestU = ytest(indices(sz_h+1:end),:);

%% Parameters
c = 1;
c2 = 1; % sigma squared
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
p = 1; s_iter =1;
X = [xtrain;xtestL];
Y = [ytrain;ytestL];
ys(:,p) = Y;
svm_model = svmtrain(Y,X,'-s 4');
[Tpred(:,p),Taccuracy(:,p),~] = svmpredict(Y, X, svm_model);
[pred(:,p),accuracy(:,p),~] = svmpredict(ytestU, xtestU, svm_model);
while p < 100
    p = p + 1;
    ynew = ytrain.*(xtrain*w) + xtrain*b;
    ynew = mean_std(ynew);
    X = [xtrain;xtestL];
    Y = [ynew;ytestL];
        
    svm_model = svmtrain(Y,X,'-s 4');
    [Tpred(:,p),Taccuracy(:,p),~] = svmpredict(Y, X, svm_model);
    [pred(:,p),accuracy(:,p),~] = svmpredict(ytestU, xtestU, svm_model);
    for t = 1:size(ytestL,1)
        ytestLs = ytestL(t,:);
        xtestLs = xtestL(t,:);
        for s = 1:size(ynew,1)
            xtrains(1,:) = xtrain(s,:);
            ytrains(1,:) = ytrain(s,:);
            ynew = ytrain.*(xtrain*w) + xtrain*b;
            ynew = mean_std(ynew);
            ynews(1,:) = ynew(s,:);
            
            %%%% Updating w and b
            for k = 1:size(w,1)
                Ltr = kernel(xtrains,xtrains,c,c2);
                Ltr2 = inv(Ltr+lam*eye(size(xtrains,1)));
                Lte = kernel(xtestLs,xtestLs,c,c2);
                Ltetr = kernel(xtestLs,xtrains,c,c2);
                Lte2 = inv(Lte+lam*eye(size(xtestLs,1)));
                
                L(:,p) = trace(Ltr2*kernel(ynews,ynews,c,c2)*Ltr2*Ltr) - 2*trace(Ltr2*kernel(ynews,ytestLs,c,c2)*Lte2*kernel(xtestLs,xtrains,c,c2));
                s_iter = s_iter+1;
                
                % first expression
                for i = 1:size(ynews,1)
                    for j = 1:size(ynews,1)
                        d_w(i,j) = (-1/c2)*(ynews(i) - ynews(j))*(ytrains(i)*I(i,k) - ytrains(j)*I(j,k));
                        d_b(i,j) = (-1/c2)*(ynews(i) - ynews(j))*(I(i,k) - I(j,k));
                    end
                end
                dl_dk = Ltr2*Ltr'*Ltr2;
                w(k) = w(k) + eta*trace(dl_dk'*[kernel(ynews,ynews,c,c2).*(d_w)]);
                b(k) = b(k)+ eta*trace(dl_dk'*[kernel(ynews,ynews,c,c2).*(d_b)]);
                
                % Second expression
                for i = 1:size(ynews,1)
                    for j = 1:size(ytestLs,1)
                        ep_w(i,j) = (-1/c2)*(ynews(i) - ytestLs(j))*ytrains(i)*I(i,k);
                        ep_b(i,j) = (-1/c2)*(ynews(i) - ytestLs(j))*I(i,k);
                    end
                end
                dl_dk2 = 2*Ltr2*Ltetr'*Lte2;
                w(k) = w(k) + eta*trace(dl_dk2'*[kernel(ynews,ytestLs,c,c2).*ep_w]);
                b(k) = b(k) + eta*trace(dl_dk2'*[kernel(ynews,ytestLs,c,c2).*ep_b]);
            end
            w = w + lamr*2*(w-1);
            b = b + lamr*2*b;
            
        end
    end
    
    
    
    storew(:,p) = w;
    storeb(:,p) = b;
    
end
figure(1);
subplot(3,1,1);
plot(L(1,1:end),'-*');
subplot(3,1,2);
plot(Taccuracy(2,1:end),'-*');
subplot(3,1,3);
plot(accuracy(2,1:end),'-*');