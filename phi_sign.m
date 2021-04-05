function s = phi_sign(x,p)
s = x;
for i = 1:length(x)
    s(i) = sign(x(i))*abs(x(i))^(p-1);
end
end

