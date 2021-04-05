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
batch = 250;
faktor_small = 0.001;
faktor = 0.1;
DoC_step = linspace(faktor_small,faktor,ranges);
p = Inf;
q = 10;

SR = zeros(1,ranges);
acc_array = zeros(2,ranges);
entropies = zeros(2,ranges);

conf_mat_25 = zeros(10,10);
conf_mat_50 = zeros(10,10);
conf_mat_75 = zeros(10,10);

num = zeros(1,n);

for i = 1:n
    out = NeuralF(w2,w3,w4,b2,b3,b4,images_temp(:,i));
    
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
DoC_faktor = faktor_small*m;
fprintf('DoC: %f\n', DoC_faktor);

[X,Xlab] = rand_sample_selector(images,labels,batch);

X = X';

[v,mu] = Singular_PQ(w2,w3,w4,b2,b3,b4,X,niter,p,q);

success2 = 0;
failure2 = 0;
success3 = 0;

for i = 1:n
    image_attack = images_temp(:,i) + DoC_faktor * v * norm(images_temp(:,i));
    out2 = NeuralF(w2,w3,w4,b2,b3,b4,image_attack);
    entropies(1,m) = entropies(1,m) + inf_entropy(out2);
    
    out3 = NeuralF(w2CE,w3CE,w4CE,b2CE,b3CE,b4CE,image_attack);
    entropies(2,m) = entropies(2,m) + inf_entropy(out3);
    
    big2 = 0;
    num2 = 0;
    
    big3 = 0;
    num3 = 0;
    
    for k = 1:10    
    
        if out2(k) > big2
            num2 = k-1;
            big2 = out2(k);
        end
        
        if out3(k) > big3
            num3 = k-1;
            big3 = out3(k);
        end

    end
    
    
    if labels(i) == num2   %For finding accuracy of the Network for FGSM-set
        success2 = success2 + 1;
    elseif ((labels(i) ~= num2) && (labels(i) == num(i))) %For finding SR of FGSM-set
        failure2 = failure2 + 1;
    end
    
    if labels(i) == num3   %For finding accuracy of the Network for FGSM-set
        success3 = success3 + 1;
    end
  
if m==25 
conf_mat_25(labels(i)+1,num2+1) = conf_mat_25(labels(i)+1,num2+1) + 1;
end

if m==50 
conf_mat_50(labels(i)+1,num2+1) = conf_mat_50(labels(i)+1,num2+1) + 1;
end

if m==75 
conf_mat_75(labels(i)+1,num2+1) = conf_mat_75(labels(i)+1,num2+1) + 1;
end
    
end
if m==25
    for j=1:10
        conf_mat_25(j,:) = round(conf_mat_25(j,:)/each_label(j),3);
    end
    
    figure(1)
    xvals = {'0','1','2','3','4','5','6','7','8','9'};
    yvals = {'0','1','2','3','4','5','6','7','8','9'};
    conf_table2 = heatmap(xvals,yvals,conf_mat_25);
    conf_table2.Title = strcat('(Inf,10)-Singular Confusion Matrix: Batch Size = 250, DoC = 0.0025');
    conf_table2.XLabel = 'Predicted Values';
    conf_table2.YLabel = 'True Values';
    export_fig(figure(1), sprintf('UNI_SingInf10_DoC25_conf.pdf'));
end
if m==50
    for j=1:10
        conf_mat_50(j,:) = round(conf_mat_50(j,:)/each_label(j),3);
    end
    
    figure(2)
    xvals = {'0','1','2','3','4','5','6','7','8','9'};
    yvals = {'0','1','2','3','4','5','6','7','8','9'};
    conf_table2 = heatmap(xvals,yvals,conf_mat_50);
    conf_table2.Title = strcat('(Inf,10)-Singular Confusion Matrix: Batch Size = 250, DoC = 0.005');
    conf_table2.XLabel = 'Predicted Values';
    conf_table2.YLabel = 'True Values';
    export_fig(figure(2), sprintf('UNI_SingInf10_DoC50_conf.pdf'));
end

if m==75
    for j=1:10
        conf_mat_75(j,:) = round(conf_mat_75(j,:)/each_label(j),3);
    end
    
    figure(3)
    xvals = {'0','1','2','3','4','5','6','7','8','9'};
    yvals = {'0','1','2','3','4','5','6','7','8','9'};
    conf_table2 = heatmap(xvals,yvals,conf_mat_75);
    conf_table2.Title = strcat('(Inf,10)-Singular Confusion Matrix: Batch Size = 250, DoC = 0.075');
    conf_table2.XLabel = 'Predicted Values';
    conf_table2.YLabel = 'True Values';
    export_fig(figure(3), sprintf('UNI_SingInf10_DoC75_conf.pdf'));
end



entropies(1,m) = entropies(1,m)/n;
entropies(2,m) = entropies(2,m)/n;

SR(1,m) = failure2/success*100;

acc_array(1,m) = success2/n*100;
acc_array(2,m) = success3/n*100;

fprintf('Accuracy for SSE DoC = %f: ', DoC_faktor);
fprintf('%f',acc_array(1,m));
disp(' %');
fprintf('Entropy for SSE DoC = %f: ', DoC_faktor);
fprintf('%f',entropies(1,m));
disp(' %');
fprintf('Accuracy for CE DoC = %f: ', DoC_faktor);
fprintf('%f',acc_array(2,m));
disp(' %');
fprintf('Entropy for CE DoC = %f: ', DoC_faktor);
fprintf('%f',entropies(2,m));
disp(' %');

end

figure(4)
hold on
h1(1) = plot(DoC_step,acc_array(1,:));
h1(2) = plot(DoC_step,acc_array(2,:));
hold off
xlabel('Degree of Change') 
ylabel('Accuracy (%)')
legend(h1, 'SSE','CE')
export_fig(figure(4), 'acc_comparison_SingInf10_uni_DoC.pdf');

figure(5)
hold on
h2(1) = plot(DoC_step,entropies(1,:));
h2(2) = plot(DoC_step,entropies(2,:));
hold off
xlabel('Degree of Change') 
ylabel('Entropy (bits)')
legend(h1, 'SSE','CE')
export_fig(figure(5), 'entropy_comparison_SingInf10_uni_DoC.pdf');

figure(6)
plot(DoC_step,SR(1,:));
xlabel('Degree of Change') 
ylabel('Success Rate (%)')
export_fig(figure(6), 'SR_comparison_SingInf10_uni_DoC.pdf');



