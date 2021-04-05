function v = CEGrad(out,label)
%Calculates the gradient of the softmax cross-entropy loss function,
%necessary for backpropagation
v = zeros(10,1);
for i = 1:10
    if (label(i) == 1)
        v(i) = softmax_CE(out,i) - 1;
    else
        v(i) = softmax_CE(out,i);
    end
end
end

