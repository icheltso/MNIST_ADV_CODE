function G = CostGradientCE(w12,w23,w34,b12,b23,b34,x,y)
%This function computes the gradient of the Softmax Cross-Entropy Loss function at a point x with
%respective label y
    J = JacobianX(w12,w23,w34,b12,b23,b34,x);
    G = transpose(J)*CEGrad(NeuralF(w12,w23,w34,b12,b23,b34,x),y);

end

