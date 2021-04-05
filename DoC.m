function d = DoC(x,xadv)
%Calculates the DoC for the 2-norm case.
    d = norm(x-xadv)/norm(x);
end

