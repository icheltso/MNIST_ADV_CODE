function J = Jmatvec(w12,w23,w34,b12,b23,b34,X,vec)
%Creates matvec functions J_i(X_b)v, where X_b is a randomly selected sample
%from the dataset.
n = size(X,2);
J = JacobianX(w12,w23,w34,b12,b23,b34,X(:,1))*vec;
for i = 2:n
    J = cat(1,J,JacobianX(w12,w23,w34,b12,b23,b34,X(:,i))*vec);
end
end