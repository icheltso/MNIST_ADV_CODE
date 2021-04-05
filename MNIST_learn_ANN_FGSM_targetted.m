load('test.mat');
labels = data_test(:,1);
y = zeros(10,10000);
for i = 1:10000
    y(labels(i)+1,i) = 1;
end

images = data_test(:,2:785);
images = images/255;

images = images';

[each_label,lab_num]=hist(labels,unique(labels));

we34 = matfile('wfour.mat');
w4 = we34.w34;
we23 = matfile('wthree.mat');
w3 = we23.w23;
we12 = matfile('wtwo.mat');
w2 = we12.w12;
bi34 = matfile('bfour.mat');
b4 = bi34.b34;
bi23 = matfile('bthree.mat');
b3 = bi23.b23;
bi12 = matfile('btwo.mat');
b2 = bi12.b12;
n = 10000;

eps_steps = linspace(0,0.255,256);
acc_array = zeros(1,256);

conf_mat_orig = zeros(10,10);
conf_mat_128 = zeros(10,10);
conf_mat_256 = zeros(10,10);

for epsn = 1:256
    counter_bin = 0;
    eps_step = eps_steps(epsn);
    success = 0;
for i = 1:n
    image_fgs = images(:,i) - eps_step * sign(CostGradient(w2,w3,w4,b2,b3,b4,images(:,i),ith_label(2)));
    image_fgs = between01(image_fgs);
    out = NeuralF(w2,w3,w4,b2,b3,b4,image_fgs);
    big = 0;
    num = 0;
for k = 1:10    %Choose most probable output
    if out(k) > big
        num = k-1;
        big = out(k);
    end
end


if labels(i) == num
    success = success + 1;
end

if epsn == 1 
conf_mat_orig(labels(i)+1,num+1) = conf_mat_orig(labels(i)+1,num+1) + 1;
end

if epsn == 128 
conf_mat_128(labels(i)+1,num+1) = conf_mat_128(labels(i)+1,num+1) + 1;
end

if epsn == 256 
conf_mat_256(labels(i)+1,num+1) = conf_mat_256(labels(i)+1,num+1) + 1;
end
    

end

acc_array(epsn) = success/n*100;
fprintf('Stepsize: %f\n',eps_step);
fprintf('Accuracy: ');
fprintf('%f',acc_array(epsn));
disp(' %');
end

for j = 1:10

conf_mat_orig(j,:) = round(conf_mat_orig(j,:)/each_label(j),3);
conf_mat_128(j,:) = round(conf_mat_128(j,:)/each_label(j),3);
conf_mat_256(j,:) = round(conf_mat_256(j,:)/each_label(j),3);

end


figure(1)
plot(eps_steps,acc_array);
xlabel('\epsilon') 
ylabel('Accuracy (%)')
export_fig(figure(1), 'pert_accv3.pdf')

figure(2)
xvals = {'0','1','2','3','4','5','6','7','8','9'};
yvals = {'0','1','2','3','4','5','6','7','8','9'};
conf_table1 = heatmap(xvals,yvals,conf_mat_orig);
conf_table1.Title = 'Original Confusion Matrix';
conf_table1.XLabel = 'Predicted Values';
conf_table1.YLabel = 'True Values';
export_fig(figure(2), 'TFGSM_orig_conf.pdf')

figure(3)
xvals = {'0','1','2','3','4','5','6','7','8','9'};
yvals = {'0','1','2','3','4','5','6','7','8','9'};
conf_table2 = heatmap(xvals,yvals,conf_mat_128);
conf_table2.Title = 'TFGSM Confusion Matrix: Targetted Value "1", \epsilon = 0.128';
conf_table2.XLabel = 'Predicted Values';
conf_table2.YLabel = 'True Values';
export_fig(figure(3), 'TFGSM_128_conf.pdf')

figure(4)
xvals = {'0','1','2','3','4','5','6','7','8','9'};
yvals = {'0','1','2','3','4','5','6','7','8','9'};
conf_table3 = heatmap(xvals,yvals,conf_mat_256);
conf_table3.Title = 'TFGSM Confusion Matrix: Targetted Value "1", \epsilon = 0.255';
conf_table3.XLabel = 'Predicted Values';
conf_table3.YLabel = 'True Values';
export_fig(figure(4), 'TFGSM_256_conf.pdf')