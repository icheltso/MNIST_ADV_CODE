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


