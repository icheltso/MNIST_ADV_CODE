function sm = softmax_CE(v,i)
%calculates the softmax function e^{v_i}/sum_{j in |v|}(e^{v_j})
expv = 0;
for j = 1:length(v)
    expv = expv + exp(v(j));
end
sm = exp(v(i))/expv;
end

