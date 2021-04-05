function F = NeuralF(w12,w23,w34,b12,b23,b34,x)
    a1 = x;
    a2 = activate(a1,w12,b12);
    a3 = activate(a2,w23,b23);
    F = activate(a3,w34,b34);
end

