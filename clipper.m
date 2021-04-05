function A2 = clipper(X,A,eps)
%Clip function, which pointwise clipse each value A_i of A to within
%[X_i-eps,X_i+eps].
%Used in Iterative FGSM and Iterative TFGSM
len = length(X);
A2 = zeros(len,1);
for i = 1:len
    A2 = min([1,X(i)+eps,max(0,X(i)-eps,A(i))]);
end

end

