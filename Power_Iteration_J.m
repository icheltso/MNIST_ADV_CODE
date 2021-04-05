function [v,mu] = Power_Iteration_J(w12,w23,w34,b12,b23,b34,X,niter)
%Power iteration algorithm for finding the dominant eigenvalue and
%associated eigenvector of the matrix J^TJ.
%Creates the universal adversarial example problem for the 2-norm case from
%Oseledets
n = size(X,1);
v = 2*rand(n,1)-ones(n,1);     %vector of uniformaly distributed random numbers in [-1,1]
    for i = 1:niter
        v1 = JTmatvec(w12,w23,w34,b12,b23,b34,X,Jmatvec(w12,w23,w34,b12,b23,b34,X,v));
        v1_norm = norm(v1);
        v = v1/v1_norm;
    end
mu = transpose(v)*JTmatvec(w12,w23,w34,b12,b23,b34,X,Jmatvec(w12,w23,w34,b12,b23,b34,X,v))/(transpose(v)*v);
end

