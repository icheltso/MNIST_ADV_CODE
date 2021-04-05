function A = iter_FGSM_MSE(w2,w3,w4,b2,b3,b4,X,Y,eps)
%Method for iterative FGSM
niter = round(255*eps);
A = X;
for i = 1:niter
    A = clipper(X,A + sign(CostGradient(w2,w3,w4,b2,b3,b4,A,Y)),eps);
end
end

