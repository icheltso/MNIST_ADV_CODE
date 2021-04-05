function y = between01(x)
    y = x;
    N = size(y);
    for i = 1:N
        if (x(i) < 0)
            y(i) = 0;
        elseif (x(i) > 1)
            y(i) = 1;
    end
end

