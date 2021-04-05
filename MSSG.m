function dCdw = MSSG(y,w,X,b)
    %Gradient of MSS
    %Take sigmoid of first gradient component
    N = length(y);
    m = length(X(:,1));
    dCdw = zeros(m,1);
    for j = 1:m
        for i = 1:N
            x = X(:,i);
            dCdw(j) = dCdw(j) - (2/N)*x(j)*(y(i) - activate(x,transpose(w),b))*activateprime(x,transpose(w),b);
        end
    end
end

