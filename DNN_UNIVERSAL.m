load('test.mat');
labels = data_test(:,1);
y = zeros(10,10000);
for i = 1:10000
    y(labels(i)+1,i) = 1;
end

images = data_test(:,2:785);
images = images/255;

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
niter = 30;
ranges = 300;
batch_sz = 10;
err_vec = zeros(1,ranges);
times = zeros(2,ranges);
steps = batch_sz*linspace(1,ranges,ranges);

for k = 1:ranges
batch = batch_sz*k;
fprintf('Batch size: %f\n', batch);

[X,Xlab] = rand_sample_selector(images,labels,batch);

X = X';

tStart1 = tic;
[v,mu] = Power_Iteration_J(w2,w3,w4,b2,b3,b4,X,niter);
times(1,k) = toc(tStart1);

m = size(X,2);

%fprintf('Dominant Iterated Eigenvalue: %f\n', mu);

tStart2 = tic;
J = JacobianX(w2,w3,w4,b2,b3,b4,X(:,1));
for i = 2:m
    J = cat(1,J,JacobianX(w2,w3,w4,b2,b3,b4,X(:,i)));
end

JT = transpose(J);

JTJ = JT*J;

[evec,eval] = eig(JTJ);

evalvec = diag(eval);
[maxeval,argmax] = max(abs(evalvec));
maxeval = evalvec(argmax);

domevec = evec(:,argmax);

times(2,k) = toc(tStart2);
%fprintf('Dominant Eigenvalue: %f\n', norm(maxeval));
err_vec(k) = min([100*norm(domevec - v)/norm(domevec),100*norm(domevec + v)/norm(domevec)]);

fprintf('Dominant Eigenvector Relative Error:');
disp(err_vec(k));
end

figure(1)
hold on
plot(steps,times(1,:));
plot(steps,times(2,:));
hold off
xlabel('Batch Size') 
ylabel('Time to Create Attack (s)')
legend('Power Iteration','Built-in Solver');
export_fig(figure(1), 'time_comparison_UNI.pdf');

figure(2)
hold on
plot(steps,err_vec);
hold off
xlabel('Batch Size') 
ylabel('Error in Dominant Eigenvector(%)')
export_fig(figure(2), 'err_comparison_UNI.pdf');


max_niter = 150;
batch = 2000;
[X,Xlab] = rand_sample_selector(images,labels,batch);
X = X';
J = JacobianX(w2,w3,w4,b2,b3,b4,X(:,1));
m = size(X,2);
for i = 2:m
    J = cat(1,J,JacobianX(w2,w3,w4,b2,b3,b4,X(:,i)));
end
JT = transpose(J);

JTJ = JT*J;

[evec,eval] = eig(JTJ);

evalvec = diag(eval);
[maxeval,argmax] = max(abs(evalvec));
maxeval = evalvec(argmax);

domevec = evec(:,argmax);

times2 = zeros(1,max_niter);
steps2 = linspace(1,max_niter,max_niter);
err_vec2 = zeros(1,max_niter);

for ntr = 1:max_niter
fprintf('Number of Iterations: %f\n', ntr);
tStart1 = tic;
[v,mu] = Power_Iteration_J(w2,w3,w4,b2,b3,b4,X,ntr);
times2(ntr) = toc(tStart1);

err_vec2(k) = min([100*norm(domevec - v)/norm(domevec),100*norm(domevec + v)/norm(domevec)]);

fprintf('Dominant Eigenvector Relative Error:');
disp(err_vec2(k));
end

figure(3)
plot(steps2,times2);
xlabel('Number of Iterations') 
ylabel('Time to Create Iterated Attack (s)')
export_fig(figure(3), 'time_comparison_niter_UNI.pdf');

figure(4)
plot(steps2,err_vec2);
xlabel('Number of Iterations') 
ylabel('Error in Dominant Eigenvector(%)')
export_fig(figure(4), 'err_comparison_niter_UNI.pdf');