function G = CostGradientFD(w12,w23,w34,b12,b23,b34,x,y)
%This function computes the gradient of the 2-norm Loss function at a point x with
%respective label y
    J = JacobianXFD(w12,w23,w34,b12,b23,b34,x);
    G = transpose(J)*(NeuralF(w12,w23,w34,b12,b23,b34,x)-y);
end
