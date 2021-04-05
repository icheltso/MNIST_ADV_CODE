function Jv = CostGradientFDv2(w12,w23,w34,b12,b23,b34,x,y)
%This function computes the gradient of the 2-norm Loss function at a point x with
%respective label y
%Calculates the Jacobian using Finite Differencing
    v = (NeuralF(w12,w23,w34,b12,b23,b34,x)-y);
    disp(size(v))
    J = zeros(10,784);
    h = 1e-8;
    a1 = x;
    disp(size(x))
    a2 = activate(a1,w12,b12);
    a3 = activate(a2,w23,b23);
    a4 = activate(a3,w34,b34);
    for i = 1:784          %Estimates a column of the Jacobian
        a1h = x + h*v;
        a2h = activate(a1h,w12,b12);
        a3h = activate(a2h,w23,b23);
        a4h = activate(a3h,w34,b34);
        J(:,i) = (a4h - a4)/h;
    end
end

