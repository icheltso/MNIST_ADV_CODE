function d = LLD(y,w,X,b) %Computes the Gradient of Log-Likelihood
    %Gradient of Log-Likelihood w.r.t. j-th weight
    %Take sigmoid of first gradient component
    v = y(1) - activate(X(:,1),transpose(w),b);
    for i=2:length(y)
        %Select x_ij
        x = X(:,i);
        %Calculate y_i - sigma(w^Tx_i + b) and concatenate this value with
        %array v
        v = cat(2,v,(y(i) - activate(x,transpose(w),b)));
    end
    %Select first sample x, dot product with v. This is the gradient w.r.t.
    %weight w_1
    d = v*transpose(X(1,:));
    for j = 2:length(w)
        %Repeat the same operation for other weights
        d = cat(1,d,v*transpose(X(j,:)));
    end
end

