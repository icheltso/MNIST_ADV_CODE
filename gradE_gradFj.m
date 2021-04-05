function v = gradE_gradFj(w12,w23,w34,b12,b23,b34,x,y,j)
    gradE = CostGradient(w12,w23,w34,b12,b23,b34,x,y);
    E = eye(10);
    gradFj = transpose(JacobianX(w12,w23,w34,b12,b23,b34,x))*E(:,j);
    v = dot(gradE,gradFj);
end

