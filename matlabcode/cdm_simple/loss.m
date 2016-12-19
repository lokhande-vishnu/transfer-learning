    Ltr = kernel(xtrain,xtrain,c,c2);
    Ltr2 = inv(Ltr+lam*eye(size(xtrain,1)));
    Lte = kernel(xtestL,xtestL,c,c2);
    Ltetr = kernel(xtestL,xtrain,c,c2);
    Lte2 = inv(Lte+lam*eye(size(xtestL,1)));

    if exist('ynew')
        disp(1);
        lossv = trace(Ltr2*kernel(ynew,ynew,c,c2)*Ltr2*Ltr) - 2*trace(Ltr2*kernel(ynew,ytestL,c,c2)*Lte2*kernel(xtestL,xtrain,c,c2));
    else
        disp(2);
        lossv = trace(Ltr2*kernel(ytrain,ytrain,c,c2)*Ltr2*Ltr) - 2*trace(Ltr2*kernel(ytrain,ytestL,c,c2)*Lte2*kernel(xtestL,xtrain,c,c2));
    end