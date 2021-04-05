function dCdb = grad_bias_MSS(y,w,X,b)
    %Gradient of MSS w.r.t. j-th weight
    %Take sigmoid of first gradient component
    N = length(y);
    dCdb = 0;
    for i = 1:N
        x = X(:,i);
        dCdb = dCdb - (2/N)*(y(i) - activate(x,transpose(w),b))*activateprime(x,transpose(w),b);
    end
end