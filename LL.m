function e = LL(y,w,X,b)
%Feed a label column vector y, a set of column vectors X, a column vector
%of weights, and a bias term b. Get Log-Likelihood.
    e = 0;
    N = length(y);
    for i=1:N
        x = X(:,i);
        e = e + (1/N)*(y(i)*log(activate(x,transpose(w),b)) + (1-y(i))*log(1 - activate(x,transpose(w),b)));
    end
end

