load('test.mat');
labels = data_test(:,1);
y = zeros(10,10000);
for i = 1:10000
    y(labels(i)+1,i) = 1;
end

images = data_test(:,2:785);
images = images/255;

images = images';

we34 = matfile('wfour.mat');
we34CE = matfile('wfourCE.mat');
w4 = we34.w34;
w4CE = we34CE.w34;
we23 = matfile('wthree.mat');
we23CE = matfile('wthreeCE.mat');
w3 = we23.w23;
w3CE = we23CE.w23;
we12 = matfile('wtwo.mat');
we12CE = matfile('wtwoCE.mat');
w2 = we12.w12;
w2CE = we12CE.w12;
bi34 = matfile('bfour.mat');
bi34CE = matfile('bfourCE.mat');
b4 = bi34.b34;
b4CE = bi34CE.b34;
bi23 = matfile('bthree.mat');
bi23CE = matfile('bthreeCE.mat');
b3 = bi23.b23;
b3CE = bi23CE.b23;
bi12 = matfile('btwo.mat');
bi12CE = matfile('btwoCE.mat');
b2 = bi12.b12;
b2CE = bi12CE.b12;
success = 0;
n = 10000;

grad_inner_prod = zeros(10,10000);
grad_inner_prod_CE = zeros(10,10000);
grad_success = 0;
grad_successCE = 0;
ll_success = 0;
ll_success_CE = 0;
jtest = [1,2,3,4,5,6,7,8,9,10];

min_inn_avg = 0;
min_inn_avg2 = 0;
max_inn_avg = 0;
max_inn_avg2 = 0;

for i = 1:n
fprintf('Started iteration %f\n', i);
out2 = activate(images(:,i),w2,b2);
out3 = activate(out2,w3,b3);
out = activate(out3,w4,b4);
for j = 1:10
    grad_inner_prod(j,i) = gradE_gradFj_normed(w2,w3,w4,b2,b3,b4,images(:,i),label_to_vector(labels(i)),j);
    grad_inner_prod_CE(j,i) = gradE_gradFj_CE_normed(w2,w3,w4,b2,b3,b4,images(:,i),label_to_vector(labels(i)),j);
end
big = 0;
num = 0;
for k = 1:10    %Choose most probable output
    if out(k) > big
        num = k-1;
        big = out(k);
    end
end

%test the hypothesis, that true label corresponds to greatest inner product
[max_inn,argmax] = max(grad_inner_prod(:,i));
[max_inn2,argmax2] = max(grad_inner_prod_CE(:,i));
max_inn_avg = max_inn_avg + max_inn;
max_inn_avg2 = max_inn_avg2 + max_inn2;
if(argmax == labels(i)+1)
    ll_success = ll_success + 1;
end
if(argmax2 == labels(i)+1)
    ll_success_CE = ll_success_CE + 1;
end

%test the hypothesis, that true label corresponds to smallest inner product
[min_inn,argmin] = min(grad_inner_prod(:,i));
min_inn_avg = min_inn_avg + min_inn;
[min_inn2,argmin2] = min(grad_inner_prod_CE(:,i));
min_inn_avg2 = min_inn_avg2 + min_inn2;
if(argmin == labels(i)+1)
    grad_success = grad_success + 1;
end
if(argmin2 == labels(i)+1)
    grad_successCE = grad_successCE + 1;
end


if labels(i) == num
    success = success + 1;
end
    

end

fprintf('Accuracy: ');
fprintf('%f',success/n*100);
disp(' %');
fprintf('p(Grad MSE coincides with -Grad F_j): ');
fprintf('%f',grad_success/n*100);
disp(' %');
fprintf('Average MSE inner product (between -1 and 1): ');
fprintf('%f\n',min_inn_avg/n);
fprintf('p(Grad CEE coincides with -Grad F_j): ');
fprintf('%f',grad_successCE/n*100);
disp(' %');
fprintf('Average CEE inner product (between -1 and 1): ');
fprintf('%f\n',min_inn_avg2/n);

fprintf('p(Grad MSE coincides with Grad of LL F_j): ');
fprintf('%f',ll_success/n*100);
disp(' %');
fprintf('Average MSE inner product (between -1 and 1): ');
fprintf('%f\n',max_inn_avg/n);
fprintf('p(Grad CEE coincides with Grad of LL F_j): ');
fprintf('%f',ll_success_CE/n*100);
disp(' %');
fprintf('Average CEE inner product (between -1 and 1): ');
fprintf('%f\n',max_inn_avg2/n);

