function v = JTmatvec(w12,w23,w34,b12,b23,b34,X,vec)
%Creates matvec functions J_i(X_b)v, where X_b is a randomly selected sample
%from the dataset.
n = size(X,2);
l = size(X,1);
m = size(w34,1);
v = zeros(l,1);
for i = 1:n
    J = JacobianX(w12,w23,w34,b12,b23,b34,X(:,i));
    v = v + transpose(J)*vec((i-1)*m+1:i*m,1);
end
end