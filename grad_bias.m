function d = grad_bias(y,w,X,b)
    d = 0;
    v = y(1) - activate(X(:,1),transpose(w),b);
    for i=2:length(y)
        x = X(:,i);
        v = cat(2,v,(y(i) - activate(x,transpose(w),b)));
    end
    d = v*ones(length(y),1);
end

