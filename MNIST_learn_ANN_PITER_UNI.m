load('test.mat');
labels = data_test(:,1);
y = zeros(10,10000);
for i = 1:10000
    y(labels(i)+1,i) = 1;
end

images = data_test(:,2:785);
images = images/255;

images_temp = images';

[each_label,lab_num]=hist(labels,unique(labels));

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
ranges = 100;
batch_sz = 30;
faktor_small = 0.001;
faktor = 0.01;
faktor2 = 0.05;
steps = batch_sz*linspace(1,ranges,ranges);

SR = zeros(3,ranges);
acc_array = zeros(3,ranges);
entropies = zeros(3,ranges);

acc_arrayCE = zeros(3,ranges);
entropiesCE = zeros(3,ranges);

conf_mat_75_p001 = zeros(10,10);
conf_mat_75_p01 = zeros(10,10);
conf_mat_75_p05 = zeros(10,10);

conf_mat_150_p001 = zeros(10,10);
conf_mat_150_p01 = zeros(10,10);
conf_mat_150_p05 = zeros(10,10);

conf_mat_300_p001 = zeros(10,10);
conf_mat_300_p01 = zeros(10,10);
conf_mat_300_p05 = zeros(10,10);

orig_class = zeros(10,n);
num = zeros(1,n);

for i = 1:n
    out = NeuralF(w2,w3,w4,b2,b3,b4,images_temp(:,i));
    orig_class(:,i) = out;
    
    big = 0;
    
    for k = 1:10    
        if out(k) > big
            num(i) = k-1;
            big = out(k);
        end
    end
    
    if labels(i) == num(i)   %For finding accuracy of the Network for FGSM-set
        success = success + 1;
    end
    
end




for m = 1:ranges
batch = batch_sz*m;
fprintf('Batch size: %f\n', batch);

[X,Xlab] = rand_sample_selector(images,labels,batch);

X = X';

[v,mu] = Power_Iteration_J(w2,w3,w4,b2,b3,b4,X,niter);

success1 = 0;
success2 = 0;
success3 = 0;
failure1 = 0;
failure2 = 0;
failure3 = 0;
success1CE = 0;
success2CE = 0;
success3CE = 0;
failure1CE = 0;
failure2CE = 0;
failure3CE = 0;

for i = 1:n
    v_small = faktor_small * v * norm(images_temp(:,i));
    v_prime = faktor * v * norm(images_temp(:,i));         %change magnitude of v so DOC is satisfied
    v_prime2 = faktor2 * v * norm(images_temp(:,i)); 
    image_attack_s = images_temp(:,i) + v_small;
    image_attack = images_temp(:,i) + v_prime;
    image_attack2 = images_temp(:,i) + v_prime2;
    out1 = NeuralF(w2,w3,w4,b2,b3,b4,image_attack_s);
    out2 = NeuralF(w2,w3,w4,b2,b3,b4,image_attack);
    out3 = NeuralF(w2,w3,w4,b2,b3,b4,image_attack2);
    out1CE = NeuralF(w2CE,w3CE,w4CE,b2CE,b3CE,b4CE,image_attack_s);
    out2CE = NeuralF(w2CE,w3CE,w4CE,b2CE,b3CE,b4CE,image_attack);
    out3CE = NeuralF(w2CE,w3CE,w4CE,b2CE,b3CE,b4CE,image_attack2);
    entropies(1,m) = entropies(1,m) + inf_entropy(out1);
    entropies(2,m) = entropies(2,m) + inf_entropy(out2);
    entropies(3,m) = entropies(3,m) + inf_entropy(out3);
    entropiesCE(1,m) = entropiesCE(1,m) + inf_entropy(out1CE);
    entropiesCE(2,m) = entropiesCE(2,m) + inf_entropy(out2CE);
    entropiesCE(3,m) = entropiesCE(3,m) + inf_entropy(out3CE);
    
    
    big1 = 0;
    big2 = 0;
    big3 = 0;
    num1 = 0;
    num2 = 0;
    num3 = 0;
    big1CE = 0;
    big2CE = 0;
    big3CE = 0;
    num1CE = 0;
    num2CE = 0;
    num3CE = 0;
    bigtrue = 0;
    numtrue = 0;
    
    for k = 1:10    
        if out1(k) > big1
            num1 = k-1;
            big1 = out1(k);
        end
    
        if out2(k) > big2
            num2 = k-1;
            big2 = out2(k);
        end
        
        if out3(k) > big3
            num3 = k-1;
            big3 = out3(k);
        end
        
        if out1CE(k) > big1CE
            num1CE = k-1;
            big1CE = out1CE(k);
        end
    
        if out2CE(k) > big2CE
            num2CE = k-1;
            big2CE = out2CE(k);
        end
        
        if out3CE(k) > big3CE
            num3CE = k-1;
            big3CE = out3CE(k);
        end

    end
    
    if labels(i) == num1   %For finding accuracy of the Network for FGSM-set
        success1 = success1 + 1;
    elseif ((labels(i) ~= num1) && (labels(i) == num(i))) %For finding SR of FGSM-set
        failure1 = failure1 + 1;
    end
    
    if labels(i) == num2   %For finding accuracy of the Network for FGSM-set
        success2 = success2 + 1;
    elseif ((labels(i) ~= num2) && (labels(i) == num(i))) %For finding SR of FGSM-set
        failure2 = failure2 + 1;
    end
    
    if labels(i) == num3   %For finding accuracy of the Network for FGSM-set
        success3 = success3 + 1;
    elseif ((labels(i) ~= num3) && (labels(i) == num(i))) %For finding SR of FGSM-set
        failure3 = failure3 + 1;
    end
    
    if labels(i) == num1CE   %For finding accuracy of the Network for FGSM-set
        success1CE = success1CE + 1;
    end
    
    if labels(i) == num2CE   %For finding accuracy of the Network for FGSM-set
        success2CE = success2CE + 1;
    end
    
    if labels(i) == num3CE   %For finding accuracy of the Network for FGSM-set
        success3CE = success3CE + 1;
    end
  
if m==25 
conf_mat_75_p001(labels(i)+1,num1+1) = conf_mat_75_p001(labels(i)+1,num1+1) + 1;
conf_mat_75_p01(labels(i)+1,num2+1) = conf_mat_75_p01(labels(i)+1,num2+1) + 1;
conf_mat_75_p05(labels(i)+1,num3+1) = conf_mat_75_p05(labels(i)+1,num3+1) + 1;
end

if m==50 
conf_mat_150_p001(labels(i)+1,num1+1) = conf_mat_150_p001(labels(i)+1,num1+1) + 1;
conf_mat_150_p01(labels(i)+1,num2+1) = conf_mat_150_p01(labels(i)+1,num2+1) + 1;
conf_mat_150_p05(labels(i)+1,num3+1) = conf_mat_150_p05(labels(i)+1,num3+1) + 1;
end

if m==100 
conf_mat_300_p001(labels(i)+1,num1+1) = conf_mat_300_p001(labels(i)+1,num1+1) + 1;
conf_mat_300_p01(labels(i)+1,num2+1) = conf_mat_300_p01(labels(i)+1,num2+1) + 1;
conf_mat_300_p05(labels(i)+1,num3+1) = conf_mat_300_p05(labels(i)+1,num3+1) + 1;
end
    
end
if m==25
    for j=1:10
        conf_mat_75_p001(j,:) = round(conf_mat_75_p001(j,:)/each_label(j),3);
        conf_mat_75_p01(j,:) = round(conf_mat_75_p01(j,:)/each_label(j),3);
        conf_mat_75_p05(j,:) = round(conf_mat_75_p05(j,:)/each_label(j),3);
    end
    
    figure(1)
    xvals = {'0','1','2','3','4','5','6','7','8','9'};
    yvals = {'0','1','2','3','4','5','6','7','8','9'};
    conf_table2 = heatmap(xvals,yvals,conf_mat_75_p001);
    conf_table2.Title = strcat('Power Iteration Confusion Matrix: Batch Size = 750, DoC = 0.001');
    conf_table2.XLabel = 'Predicted Values';
    conf_table2.YLabel = 'True Values';
    export_fig(1), sprintf('UNI_750_001_conf.pdf');

    figure(2)
    xvals = {'0','1','2','3','4','5','6','7','8','9'};
    yvals = {'0','1','2','3','4','5','6','7','8','9'};
    conf_table3 = heatmap(xvals,yvals,conf_mat_75_p01);
    conf_table3.Title = strcat('Power Iteration Confusion Matrix: Batch Size = 750, DoC = 0.01');
    conf_table3.XLabel = 'Predicted Values';
    conf_table3.YLabel = 'True Values';
    export_fig(2), sprintf('UNI_750_01_conf.pdf');
    
    figure(3)
    xvals = {'0','1','2','3','4','5','6','7','8','9'};
    yvals = {'0','1','2','3','4','5','6','7','8','9'};
    conf_table3 = heatmap(xvals,yvals,conf_mat_75_p05);
    conf_table3.Title = strcat('Power Iteration Confusion Matrix: Batch Size = 750, DoC = 0.05');
    conf_table3.XLabel = 'Predicted Values';
    conf_table3.YLabel = 'True Values';
    export_fig(3), sprintf('UNI_750_05_conf.pdf');
end
if m==50
    for j=1:10
        conf_mat_150_p001(j,:) = round(conf_mat_150_p001(j,:)/each_label(j),3);
        conf_mat_150_p01(j,:) = round(conf_mat_150_p01(j,:)/each_label(j),3);
        conf_mat_150_p05(j,:) = round(conf_mat_150_p05(j,:)/each_label(j),3);
    end
    
    figure(4)
    xvals = {'0','1','2','3','4','5','6','7','8','9'};
    yvals = {'0','1','2','3','4','5','6','7','8','9'};
    conf_table2 = heatmap(xvals,yvals,conf_mat_150_p001);
    conf_table2.Title = strcat('Power Iteration Confusion Matrix: Batch Size = 1500, DoC = 0.001');
    conf_table2.XLabel = 'Predicted Values';
    conf_table2.YLabel = 'True Values';
    export_fig(4), sprintf('UNI_1500_001_conf.pdf');

    figure(5)
    xvals = {'0','1','2','3','4','5','6','7','8','9'};
    yvals = {'0','1','2','3','4','5','6','7','8','9'};
    conf_table3 = heatmap(xvals,yvals,conf_mat_150_p01);
    conf_table3.Title = strcat('Power Iteration Confusion Matrix: Batch Size = 1500, DoC = 0.01');
    conf_table3.XLabel = 'Predicted Values';
    conf_table3.YLabel = 'True Values';
    export_fig(5), sprintf('UNI_1500_01_conf.pdf');
    
    figure(6)
    xvals = {'0','1','2','3','4','5','6','7','8','9'};
    yvals = {'0','1','2','3','4','5','6','7','8','9'};
    conf_table3 = heatmap(xvals,yvals,conf_mat_150_p05);
    conf_table3.Title = strcat('Power Iteration Confusion Matrix: Batch Size = 1500, DoC = 0.05');
    conf_table3.XLabel = 'Predicted Values';
    conf_table3.YLabel = 'True Values';
    export_fig(6), sprintf('UNI_1500_05_conf.pdf');
end

if m==100
    for j=1:10
        conf_mat_300_p001(j,:) = round(conf_mat_300_p001(j,:)/each_label(j),3);
        conf_mat_300_p01(j,:) = round(conf_mat_300_p01(j,:)/each_label(j),3);
        conf_mat_300_p05(j,:) = round(conf_mat_300_p05(j,:)/each_label(j),3);
    end
    
    figure(7)
    xvals = {'0','1','2','3','4','5','6','7','8','9'};
    yvals = {'0','1','2','3','4','5','6','7','8','9'};
    conf_table2 = heatmap(xvals,yvals,conf_mat_300_p001);
    conf_table2.Title = strcat('Power Iteration Confusion Matrix: Batch Size = 3000, DoC = 0.001');
    conf_table2.XLabel = 'Predicted Values';
    conf_table2.YLabel = 'True Values';
    export_fig(7), sprintf('UNI_3000_001_conf.pdf');

    figure(8)
    xvals = {'0','1','2','3','4','5','6','7','8','9'};
    yvals = {'0','1','2','3','4','5','6','7','8','9'};
    conf_table3 = heatmap(xvals,yvals,conf_mat_300_p01);
    conf_table3.Title = strcat('Power Iteration Confusion Matrix: Batch Size = 3000, DoC = 0.01');
    conf_table3.XLabel = 'Predicted Values';
    conf_table3.YLabel = 'True Values';
    export_fig(8), sprintf('UNI_3000_01_conf.pdf');
    
    figure(9)
    xvals = {'0','1','2','3','4','5','6','7','8','9'};
    yvals = {'0','1','2','3','4','5','6','7','8','9'};
    conf_table3 = heatmap(xvals,yvals,conf_mat_300_p05);
    conf_table3.Title = strcat('Power Iteration Confusion Matrix: Batch Size = 3000, DoC = 0.05');
    conf_table3.XLabel = 'Predicted Values';
    conf_table3.YLabel = 'True Values';
    export_fig(9), sprintf('UNI_3000_05_conf.pdf');
end



entropies(1,m) = entropies(1,m)/n;
entropies(2,m) = entropies(2,m)/n;
entropies(3,m) = entropies(3,m)/n;

entropiesCE(1,m) = entropiesCE(1,m)/n;
entropiesCE(2,m) = entropiesCE(2,m)/n;
entropiesCE(3,m) = entropiesCE(3,m)/n;

SR(1,m) = failure1/success*100;
SR(2,m) = failure2/success*100;
SR(3,m) = failure3/success*100;

acc_array(1,m) = success1/n*100;
acc_array(2,m) = success2/n*100;
acc_array(3,m) = success3/n*100;

acc_arrayCE(1,m) = success1CE/n*100;
acc_arrayCE(2,m) = success2CE/n*100;
acc_arrayCE(3,m) = success3CE/n*100;

fprintf('Accuracy SSE 0.001: ');
fprintf('%f',acc_array(1,m));
disp(' %');
fprintf('Accuracy SSE 0.01: ');
fprintf('%f',acc_array(2,m));
disp(' %');
fprintf('Accuracy SSE 0.05: ');
fprintf('%f',acc_array(3,m));
disp(' %');
fprintf('Accuracy CE 0.001: ');
fprintf('%f',acc_arrayCE(1,m));
disp(' %');
fprintf('Accuracy CE 0.01: ');
fprintf('%f',acc_arrayCE(2,m));
disp(' %');
fprintf('Accuracy CE 0.05: ');
fprintf('%f',acc_arrayCE(3,m));
disp(' %');

disp(' %');

end

figure(10)
hold on
h1(1) = plot(steps,acc_array(1,:));
h1(2) = plot(steps,acc_array(2,:));
h1(3) = plot(steps,acc_array(3,:));
h1(4) = plot(steps,acc_arrayCE(1,:));
h1(5) = plot(steps,acc_arrayCE(2,:));
h1(6) = plot(steps,acc_arrayCE(3,:));
hold off
xlabel('Batch Size') 
ylabel('Accuracy (%)')
legend(h1, 'SSE, DoC = 0.001','SSE, DoC = 0.01','SSE, DoC = 0.05','CE, DoC = 0.001','CE, DoC = 0.01','CE, DoC = 0.05','Location','east','NumColumns',1);
export_fig(figure(10), 'acc_comparison_piter_uni.pdf');

figure(11)
hold on
h2(1) = plot(steps,entropies(1,:));
h2(2) = plot(steps,entropies(2,:));
h2(3) = plot(steps,entropies(3,:));
h2(4) = plot(steps,entropiesCE(1,:));
h2(5) = plot(steps,entropiesCE(2,:));
h2(6) = plot(steps,entropiesCE(3,:));
hold off
xlabel('Batch Size') 
ylabel('Entropy (bits)')
legend(h2, 'SSE, DoC = 0.001','SSE, DoC = 0.01','SSE, DoC = 0.05','CE, DoC = 0.001','CE, DoC = 0.01','CE, DoC = 0.05','Location','east','NumColumns',1);
export_fig(figure(11), 'entropy_comparison_piter_uni.pdf');

figure(12)
hold on
h2(1) = plot(steps,SR(1,:));
h2(2) = plot(steps,SR(2,:));
h2(3) = plot(steps,SR(3,:));
hold off
xlabel('Batch Size') 
ylabel('Success Rate (%)')
legend(h2, 'DoC = 0.001','DoC = 0.01','DoC = 0.05','Location','east','NumColumns',1);
export_fig(figure(12), 'SR_comparison_piter_uni.pdf');



