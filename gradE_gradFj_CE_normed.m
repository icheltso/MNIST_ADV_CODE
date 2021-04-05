function v = gradE_gradFj_CE_normed(w12,w23,w34,b12,b23,b34,x,y,j)
    %Function for finding the inner product between Gradient of Cross
    %entropy loss function and j-th row of jacobian
    gradE = CostGradientCE(w12,w23,w34,b12,b23,b34,x,y);
    gradE = gradE/norm(gradE);
    E = eye(10);
    gradFj = transpose(JacobianX(w12,w23,w34,b12,b23,b34,x))*E(:,j);
    gradFj = gradFj/norm(gradFj);
    v = dot(gradE,gradFj);
end

