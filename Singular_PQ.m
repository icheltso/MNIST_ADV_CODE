function [v,s] = Singular_PQ(w12,w23,w34,b12,b23,b34,X,niter,p,q)
%Power method for finding the (p,q)-singular vectors of the matrix J^TJ.
%Creates the universal adversarial example problem for the (p,q)-norm case from
%Oseledets
if p == Inf
    p_prime = 1;
elseif p == 1
    p_prime = Inf;
else
    p_prime = p/(p-1);
end
n = size(X,1);
v = 2*rand(n,1)-ones(n,1);     %vector of uniformaly distributed random numbers in [-1,1]
v = v/norm(v,p);
s = norm(Jmatvec(w12,w23,w34,b12,b23,b34,X,v),q);
    for i = 1:niter
        phi1 = phi_sign(Jmatvec(w12,w23,w34,b12,b23,b34,X,v),q);
        Sx = phi_sign(JTmatvec(w12,w23,w34,b12,b23,b34,X,phi1),p_prime);
        v = Sx/norm(Sx,p);
        s = norm(Jmatvec(w12,w23,w34,b12,b23,b34,X,v),q);
    end
end