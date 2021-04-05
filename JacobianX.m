function J = JacobianX(w12,w23,w34,b12,b23,b34,x)
    a1 = x;
    a2 = activate(a1,w12,b12);
    a3 = activate(a2,w23,b23);
    D1 = diag(activateprime(a1,w12,b12));
    D2 = diag(activateprime(a2,w23,b23));
    D3 = diag(activateprime(a3,w34,b34));
    J = D3*w34*D2*w23*D1*w12;
end
