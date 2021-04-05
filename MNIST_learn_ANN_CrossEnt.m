load('train.mat');
labels = data(:,1);

y = zeros(10,60000); %Correct outputs vector
for i = 1:60000
    y(labels(i)+1,i) = 1;
end

images = data(:,2:785);
images = images/255;   %Standartization

images = images'; %Input vectors

hn1 = 90; %Number of neurons in the first hidden layer
hn2 = 60; %Number of neurons in the second hidden layer

%Initialize weights as standard normal dists
%w12 = randn(hn1,784)*sqrt(2/784);
%w23 = randn(hn2,hn1)*sqrt(2/hn1);
%w34 = randn(10,hn2)*sqrt(2/hn2);
w12 = unifrnd(-sqrt(6)/sqrt(784+hn1),sqrt(6)/sqrt(784+hn1),[hn1,784]);
w23 = unifrnd(-sqrt(6)/sqrt(hn1+hn2),sqrt(6)/sqrt(hn1+hn2),[hn2,hn1]);
w34 = unifrnd(-sqrt(6)/sqrt(hn2+10),sqrt(6)/sqrt(hn2+10),[10,hn2]);
%b12 = randn(hn1,1);
%b23 = randn(hn2,1);
%b34 = randn(10,1);
b12 = zeros(hn1,1);
b23 = zeros(hn2,1);
b34 = zeros(10,1);

%Set learning rate
eta = 0.0055;

%Initialize error and grad vectors
error4 = zeros(10,1);
error3 = zeros(hn2,1);
error2 = zeros(hn1,1);
errortot4 = zeros(10,1);
errortot3 = zeros(hn2,1);
errortot2 = zeros(hn1,1);
grad4 = zeros(10,1);
grad3 = zeros(hn2,1);
grad2 = zeros(hn1,1);

epochs = 60;

m = 10; %Minibatch size

for k = 1:epochs %Outer epoch loop
    
    batches = 1;
    
    for j = 1:60000/m
        error4 = zeros(10,1);
        error3 = zeros(hn2,1);
        error2 = zeros(hn1,1);
        errortot4 = zeros(10,1);
        errortot3 = zeros(hn2,1);
        errortot2 = zeros(hn1,1);
        grad4 = zeros(10,1);
        grad3 = zeros(hn2,1);
        grad2 = zeros(hn1,1);
    for i = batches:batches+m-1 %Loop over each minibatch
    
    %Feed forward
    a1 = images(:,i);
    a2 = activate(a1,w12,b12);
    a3 = activate(a2,w23,b23);
    a4 = activate(a3,w34,b34); %Output vector
    
    %backpropagation
    error4 = CEGrad(a4,y(:,i)).*activateprime(a3,w34,b34);     %Del C * sigma' for Cross-Entropy
    error3 = (w34'*error4).*activateprime(a2,w23,b23);
    error2 = (w23'*error3).*activateprime(a1,w12,b12);
    
    errortot4 = errortot4 + error4;
    errortot3 = errortot3 + error3;
    errortot2 = errortot2 + error2;
    grad4 = grad4 + error4*a3';
    grad3 = grad3 + error3*a2';
    grad2 = grad2 + error2*a1';

    end
    
    %Gradient descent - Update Weights at every layer
    w34 = w34 - eta/m*grad4;
    w23 = w23 - eta/m*grad3;
    w12 = w12 - eta/m*grad2;
    b34 = b34 - eta/m*errortot4;
    b23 = b23 - eta/m*errortot3;
    b12 = b12 - eta/m*errortot2;
    
    batches = batches + m;
    
    end
    fprintf('Epochs:');
    disp(k) %Track number of epochs
    [images,y] = rand_sample_selector(images',y',size(images',1)); %Shuffles order of the images for next epoch
    images = images';
    y = y';
end

disp('Training done!')
%Saves the parameters
save('wfourCE.mat','w34');
save('wthreeCE.mat','w23');
save('wtwoCE.mat','w12');
save('bfourCE.mat','b34');
save('bthreeCE.mat','b23');
save('btwoCE.mat','b12');