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

eps_steps = linspace(0,0.127,128);
acc_array = zeros(2,128);
SR = zeros(2,128);
avg_time = zeros(2,128);
out_true = zeros(10,10000);
big_true = zeros(1,10000);
num_true = zeros(1,10000);
neural_correct = 0;
entropies = zeros(2,128);
deg_change = zeros(2,128);

conf_mat_orig = zeros(10,10);
conf_mat_128 = zeros(10,10);
conf_mat_fd_128 = zeros(10,10);

for epsn = 1:128
    counter_bin = 0;
    eps_step = eps_steps(epsn);
    success = 0;
    success_fd = 0;
    failure = 0;
    failure_fd = 0;
    tStart = tic;
for i = 1:n
    tic
    image_fgs = images(:,i) + eps_step * sign(CostGradient(w2,w3,w4,b2,b3,b4,images(:,i),label_to_vector(labels(i))));
    avg_time(1,epsn) = avg_time(1,epsn) + toc;
    tic
    image_fgs_fd = images(:,i) + eps_step * sign(CostGradientFD(w2,w3,w4,b2,b3,b4,images(:,i),label_to_vector(labels(i))));
    avg_time(2,epsn) = avg_time(1,epsn) + toc;
    image_fgs = between01(image_fgs);
    image_fgs_fd = between01(image_fgs_fd);
    out = NeuralF(w2,w3,w4,b2,b3,b4,image_fgs);
    entropies(1,epsn) = entropies(1,epsn) + inf_entropy(out);
    deg_change(1,epsn) = deg_change(1,epsn) + DoC(images(:,i),image_fgs);
    out_fd = NeuralF(w2,w3,w4,b2,b3,b4,image_fgs_fd);
    entropies(2,epsn) = entropies(2,epsn) + inf_entropy(out_fd);
    deg_change(2,epsn) = deg_change(2,epsn) + DoC(images(:,i),image_fgs_fd);
    
    if (epsn == 1)   %Calculate the labels of unperturbed images on iteration 1 (eps = 0)
        out_true(:,i) = NeuralF(w2,w3,w4,b2,b3,b4,images(:,i));
        for k = 1:10    %Choose most probable output for FGSM    
            if out_true(k,i) > big_true(i) %Choose most probable output for original
                num_true(i) = k-1;
                big_true(i) = out_true(k,i);
            end
        end
        if labels(i) == num_true(i)
            neural_correct = neural_correct+1;
        end
    end
    big = 0;
    big_fd = 0;
    num = 0;
    num_fd = 0;
for k = 1:10    
    if out(k) > big       %Choose most probable output for FGSM
        num = k-1;
        big = out(k);
    end
    
    if out_fd(k) > big_fd %Choose most probable output for FD-FGSM
        num_fd = k-1;
        big_fd = out_fd(k);
    end
end


if labels(i) == num   %For finding accuracy of the Network for FGSM-set
    success = success + 1;
elseif ((labels(i) ~= num) && (labels(i) == num_true(i))) %For finding SR of FGSM-set
    failure = failure + 1;
end

if labels(i) == num_fd %For finding accuracy of the Network for FD-FGSM-set
    success_fd = success_fd + 1;
elseif ((labels(i) ~= num_fd) && (labels(i) == num_true(i))) %For finding SR of FD-FGSM-set
    failure_fd = failure_fd + 1;
end

if epsn == 1 
conf_mat_orig(labels(i)+1,num_true(i)+1) = conf_mat_orig(labels(i)+1,num_true(i)+1) + 1;
end

if epsn == 128 
conf_mat_128(labels(i)+1,num+1) = conf_mat_128(labels(i)+1,num+1) + 1;
conf_mat_fd_128(labels(i)+1,num_fd+1) = conf_mat_fd_128(labels(i)+1,num_fd+1) + 1;
end
    

end
tEnd = toc(tStart);
avg_time(1,epsn) = avg_time(1,epsn)/n;
avg_time(2,epsn) = avg_time(2,epsn)/n;
entropies(1,epsn) = entropies(1,epsn)/n;
entropies(2,epsn) = entropies(2,epsn)/n;
deg_change(1,epsn) = deg_change(1,epsn)/n;
deg_change(2,epsn) = deg_change(2,epsn)/n;

SR(1,epsn) = failure/neural_correct*100;
acc_array(1,epsn) = success/n*100;
SR(2,epsn) = failure_fd/neural_correct*100;
acc_array(2,epsn) = success_fd/n*100;
fprintf('Stepsize: %f\n',eps_step);
fprintf('Time to parse through entire test set: %f s\n', tEnd);
fprintf('Accuracy FGSM: ');
fprintf('%f',acc_array(1,epsn));
disp(' %');
fprintf('Success Rate FGSM: ');
fprintf('%f',SR(1,epsn));
disp(' %');
fprintf('Accuracy FD-FGSM: ');
fprintf('%f',acc_array(2,epsn));
disp(' %');
fprintf('Success Rate FD-FGSM: ');
fprintf('%f',SR(2,epsn));
disp(' %');
end

for j = 1:10

conf_mat_orig(j,:) = round(conf_mat_orig(j,:)/each_label(j),3);
conf_mat_128(j,:) = round(conf_mat_128(j,:)/each_label(j),3);
conf_mat_fd_128(j,:) = round(conf_mat_fd_128(j,:)/each_label(j),3);

end


figure(1)
plot(eps_steps,acc_array(1,:));
xlabel('\epsilon') 
ylabel('Accuracy (%)')
export_fig(figure(1), 'pert_acc_FGSM128.pdf')

figure(2)
plot(eps_steps,acc_array(2,:));
xlabel('\epsilon') 
ylabel('Accuracy (%)')
export_fig(figure(2), 'pert_acc_FGSM_fd128.pdf')

figure(3)
plot(eps_steps,acc_array(1,:));
xlabel('\epsilon') 
ylabel('Success Rate (%)')
export_fig(figure(3), 'pert_sr_FGSM128.pdf')

figure(4)
plot(eps_steps,acc_array(2,:));
xlabel('\epsilon') 
ylabel('Success Rate (%)')
export_fig(figure(4), 'pert_sr_FGSM_fd128.pdf')

figure(5)
xvals = {'0','1','2','3','4','5','6','7','8','9'};
yvals = {'0','1','2','3','4','5','6','7','8','9'};
conf_table1 = heatmap(xvals,yvals,conf_mat_orig);
conf_table1.Title = 'Original Confusion Matrix';
conf_table1.XLabel = 'Predicted Values';
conf_table1.YLabel = 'True Values';
export_fig(figure(5), 'FGSM_orig_conf.pdf')

figure(6)
xvals = {'0','1','2','3','4','5','6','7','8','9'};
yvals = {'0','1','2','3','4','5','6','7','8','9'};
conf_table2 = heatmap(xvals,yvals,conf_mat_128);
conf_table2.Title = 'FGSM Confusion Matrix: \epsilon = 0.128';
conf_table2.XLabel = 'Predicted Values';
conf_table2.YLabel = 'True Values';
export_fig(figure(6), 'FGSM_128_conf.pdf')

figure(7)
xvals = {'0','1','2','3','4','5','6','7','8','9'};
yvals = {'0','1','2','3','4','5','6','7','8','9'};
conf_table2 = heatmap(xvals,yvals,conf_mat_fd_128);
conf_table2.Title = 'FGSM Confusion Matrix: \epsilon = 0.128';
conf_table2.XLabel = 'Predicted Values';
conf_table2.YLabel = 'True Values';
export_fig(figure(7), 'FGSM_128_fd_conf.pdf')

figure(8)
hold on
plot(eps_steps,acc_array(1,:));
plot(eps_steps,acc_array(2,:));
hold off
xlabel('\epsilon') 
ylabel('Accuracy (%)')
legend('FGSM','FD-FGSM');
export_fig(figure(8), 'acc_comparison_BvW.pdf');

figure(9)
hold on
plot(eps_steps,SR(1,:));
plot(eps_steps,SR(2,:));
hold off
xlabel('\epsilon') 
ylabel('Success Rate (%)')
legend('FGSM','FD-FGSM');
export_fig(figure(9), 'sr_comparison_BvW.pdf');

figure(10)
hold on
plot(eps_steps,avg_time(1,:));
plot(eps_steps,avg_time(2,:));
hold off
xlabel('\epsilon') 
ylabel('Average Time to Create Attack (s)')
legend('FGSM','FD-FGSM');
export_fig(figure(10), 'time_comparison_BvW.pdf');

figure(11)
hold on
plot(eps_steps,entropies(1,:));
plot(eps_steps,entropies(2,:));
hold off
xlabel('\epsilon') 
ylabel('Entropy (bits)')
legend('FGSM','FD-FGSM');
export_fig(figure(11), 'entropy_comparison_BvW.pdf');

figure(12)
hold on
plot(eps_steps,deg_change(1,:));
plot(eps_steps,deg_change(2,:));
hold off
xlabel('\epsilon') 
ylabel('DoC')
legend('FGSM','FD-FGSM');
export_fig(figure(12), 'doc_comparison_BvW.pdf');