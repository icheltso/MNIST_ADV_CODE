function [w,b] = GradDescMNISTBinary_LogReg(iter,step,y,X)
    %All inputs are column vectors (with exception of iter and b), X is a matrix of column vectors.
    dim = size(X,1);
    w = sqrt(1/dim)*randn(dim,1);
    b = 1;
    for i = 1:iter
        LogLik = LL(y,w,X,b);
        disp(LogLik)
        grad = LLD(y,w,X,b);
        grad_b = grad_bias(y,w,X,b);
        w = w + step*grad;
        b = b + step*grad_b;
    end
end

