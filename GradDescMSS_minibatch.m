function [w,b] = GradDescMSS_minibatch(iter,step,y,X,batch)
    %All inputs are column vectors (with exception of iter and b), X is a matrix of column vectors.
    dim = size(X,1);
    w = sqrt(1/dim)*randn(dim,1);
    b = 0;
    for i = 1:iter
        MSE_true = MSS(y,w,X,b);
        disp(MSE_true)
        [randX,randy] = rand_sample_selector(transpose(X),y,batch);
        randX2 = transpose(randX);
        MSE = MSS(randy,w,randX2,b);
        grad = MSSG(randy,w,randX2,b);
        grad_b = grad_bias_MSS(randy,w,randX2,b);
        w = w - step*grad;
        b = b - step*grad_b;
    end
end
