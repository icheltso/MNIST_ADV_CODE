function [B,Blab] = rand_sample_selector(X,lab,n)
%Takes X array of sample-row vectors, The corresponding labels 'lab' and
%'n' samples to be selected. Returns samples that were selected.
    c = randperm(size(X, 1)); 
    c = c(1:n);  
    B=X(c,:);
    Blab = lab(c,:);
end

