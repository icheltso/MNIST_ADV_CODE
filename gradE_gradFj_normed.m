function v = gradE_gradFj_normed(w12,w23,w34,b12,b23,b34,x,y,j)
    gradE = CostGradient(w12,w23,w34,b12,b23,b34,x,y);
    gradE = gradE/norm(gradE);
    E = eye(10);
    gradFj = transpose(JacobianX(w12,w23,w34,b12,b23,b34,x))*E(:,j);
    gradFj = gradFj/norm(gradFj);
    v = dot(gradE,gradFj);
end

