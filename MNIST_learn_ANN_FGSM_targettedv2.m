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
n = 10000;

SR = zeros(11,256);
eps_steps = linspace(0,0.255,256);
acc_array = zeros(11,256);
acc_arrayCE = zeros(11,256);
entropies = zeros(11,256);
entropiesCE = zeros(11,256);
orig_entropy = zeros(1,256);
orig_entropyCE = zeros(1,256);
deg_change = zeros(11,256);
deg_changeCE = zeros(11,256);

ent_correct_guess_SSE = zeros(11,256);
ent_incorrect_guess_SSE = zeros(11,256);
ent_correct_guess_CE = zeros(11,256);
ent_incorrect_guess_CE = zeros(11,256);


for tc = 1:11
    fprintf('Begin Target Class %f\n',tc-1);
    conf_mat_orig = zeros(10,10);
    conf_mat_128 = zeros(10,10);
    conf_mat_256 = zeros(10,10);
for epsn = 1:256
    counter_bin = 0;
    eps_step = eps_steps(epsn);
    success = 0;
    failure = 0;
for i = 1:n
    if (tc == 11)
        image_fgs = images(:,i) + eps_step * sign(CostGradient(w2,w3,w4,b2,b3,b4,images(:,i),label_to_vector(labels(i))));
        image_fgs_CE = images(:,i) + eps_step * sign(CostGradientCE(w2CE,w3CE,w4CE,b2CE,b3CE,b4CE,images(:,i),label_to_vector(labels(i))));
    else
        image_fgs = images(:,i) - eps_step * sign(CostGradient(w2,w3,w4,b2,b3,b4,images(:,i),ith_label(tc)));
        image_fgs_CE = images(:,i) - eps_step * sign(CostGradientCE(w2CE,w3CE,w4CE,b2CE,b3CE,b4CE,images(:,i),ith_label(tc)));
    end
    image_fgs = between01(image_fgs);
    image_fgs_CE = between01(image_fgs_CE);
    out = NeuralF(w2,w3,w4,b2,b3,b4,image_fgs);
    outCE = NeuralF(w2CE,w3CE,w4CE,b2CE,b3CE,b4CE,image_fgs_CE);
    out_true = NeuralF(w2,w3,w4,b2,b3,b4,images(:,i));
    out_trueCE = NeuralF(w2CE,w3CE,w4CE,b2CE,b3CE,b4CE,images(:,i));
    entropies(tc,epsn) = entropies(tc,epsn) + inf_entropy(out);
    entropiesCE(tc,epsn) = entropiesCE(tc,epsn) + inf_entropy(outCE);
    deg_change(tc,epsn) = deg_change(tc,epsn) + DoC(images(:,i),image_fgs);
    deg_changeCE(tc,epsn) = deg_changeCE(tc,epsn) + DoC(images(:,i),image_fgs_CE);
    orig_entropy(epsn) = orig_entropy(epsn) + inf_entropy(out_true);
    orig_entropyCE(epsn) = orig_entropyCE(epsn) + inf_entropy(out_true);
    big = 0;
    num = 0;
    big_true = 0;
    num_true = 0;
for k = 1:10    %Choose most probable output for Adv. ex.
    if out(k) > big
        num = k-1;
        big = out(k);
    end
    if out_true(k) > big_true %Choose most probable output for original
        num_true = k-1;
        big_true = out_true(k);
    end
end


if labels(i) == num
    success = success + 1;
    ent_correct_guess_SSE(tc,epsn) = ent_correct_guess_SSE(tc,epsn) + inf_entropy(out);
elseif (labels(i) ~= num)
    ent_incorrect_guess_SSE(tc,epsn) = ent_incorrect_guess_SSE(tc,epsn) + inf_entropy(out);
elseif (labels(i) ~= num) && (labels(i) == num_true)
    failure = failure + 1;
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

ent_correct_guess_SSE(tc,epsn) = ent_correct_guess_SSE(tc,epsn)/success;
ent_incorrect_guess_SSE(tc,epsn) = ent_incorrect_guess_SSE(tc,epsn)/(n-success);

entropies(tc,epsn) = entropies(tc,epsn)/n;
orig_entropy(epsn) = orig_entropy(epsn)/n;
deg_change(tc,epsn) = deg_change(tc,epsn)/n;

SR(tc,epsn) = failure/9199*100;
acc_array(tc,epsn) = success/n*100;
fprintf('Stepsize: %f\n',eps_step);
fprintf('Accuracy: ');
fprintf('%f',acc_array(tc,epsn));
disp(' %');
fprintf('Success Rate: ');
fprintf('%f',SR(tc,epsn));
disp(' %');

end

for j = 1:10

conf_mat_128(j,:) = round(conf_mat_128(j,:)/each_label(j),3);
conf_mat_256(j,:) = round(conf_mat_256(j,:)/each_label(j),3);

end


if (tc ~= 11)
    
    figure(2*(tc-1) + 1)
    xvals = {'0','1','2','3','4','5','6','7','8','9'};
    yvals = {'0','1','2','3','4','5','6','7','8','9'};
    conf_table2 = heatmap(xvals,yvals,conf_mat_128);
    conf_table2.Title = strcat('TFGSM Confusion Matrix: \epsilon = 0.128, Target Class = ',num2str(tc-1));
    conf_table2.XLabel = 'Predicted Values';
    conf_table2.YLabel = 'True Values';
    export_fig(figure(2*(tc-1) + 1), sprintf('TFGSM_128_conf_%d.pdf', tc-1));

    figure(2*(tc-1) + 2)
    xvals = {'0','1','2','3','4','5','6','7','8','9'};
    yvals = {'0','1','2','3','4','5','6','7','8','9'};
    conf_table3 = heatmap(xvals,yvals,conf_mat_256);
    conf_table3.Title = strcat('TFGSM Confusion Matrix: \epsilon = 0.255, Target Class = ',num2str(tc-1));
    conf_table3.XLabel = 'Predicted Values';
    conf_table3.YLabel = 'True Values';
    export_fig(figure(2*(tc-1) + 2), sprintf('TFGSM_256_conf_%d.pdf',tc-1));
end

end

C = distinguishable_colors(11);

figure(21)
hold on
for k=1:11
    h1(k) = plot(eps_steps,acc_array(k,:));
end
hold off
xlabel('\epsilon') 
ylabel('Accuracy (%)')
legend(h1, 'Target Class 0','Target Class 1','Target Class 2','Target Class 3','Target Class 4','Target Class 5','Target Class 6','Target Class 7','Target Class 8','Target Class 9', 'Untargetted FGSM', 'Location','northeast','NumColumns',1);
export_fig(figure(21), 'TFGSM_acc_comparison.pdf');


figure(22)
hold on
for k=1:11
    h2(k) = plot(eps_steps,SR(k,:));
end
hold off
xlabel('\epsilon') 
ylabel('Success Rate (%)')
legend(h2, 'Target Class 0','Target Class 1','Target Class 2','Target Class 3','Target Class 4','Target Class 5','Target Class 6','Target Class 7','Target Class 8','Target Class 9', 'Untargetted FGSM', 'Location','southeast','NumColumns',1);
export_fig(figure(22), 'TFGSM_SR_comparison.pdf');

figure(23)
hold on
for k=1:11
    h3(k) = plot(eps_steps,entropies(k,:),'color',C(k,:));
end
hold off
xlabel('\epsilon') 
ylabel('Entropy (bits)')
legend(h3, 'Target Class 0','Target Class 1','Target Class 2','Target Class 3','Target Class 4','Target Class 5','Target Class 6','Target Class 7','Target Class 8','Target Class 9', 'Untargetted FGSM', 'Location','south','NumColumns',2);
export_fig(figure(23), 'TFGSM_entropy_comparison.pdf');

figure(24)
hold on
for k=1:11
    h4(k) = plot(eps_steps,deg_change(k,:));
end
hold off
xlabel('\epsilon') 
ylabel('Degree of Change')
legend(h4, 'Target Class 0','Target Class 1','Target Class 2','Target Class 3','Target Class 4','Target Class 5','Target Class 6','Target Class 7','Target Class 8','Target Class 9', 'Untargetted FGSM','Location','southeast','NumColumns',1);
export_fig(figure(24), 'TFGSM_DoC_comparison.pdf');

figure(25)
hold on
for k=1:11
    h5(k) = plot(eps_steps,ent_correct_guess_SSE(k,:));
end
hold off
xlabel('\epsilon') 
ylabel('Entropy of Correct Predictions (bits)')
legend(h5, 'Target Class 0','Target Class 1','Target Class 2','Target Class 3','Target Class 4','Target Class 5','Target Class 6','Target Class 7','Target Class 8','Target Class 9', 'Untargetted FGSM','Location','southeast','NumColumns',1);
export_fig(figure(25), 'TFGSM_CorEnt_comparison.pdf');

figure(26)
hold on
for k=1:11
    h6(k) = plot(eps_steps,ent_incorrect_guess_SSE(k,:));
end
hold off
xlabel('\epsilon') 
ylabel('Entropy of Incorrect Predictions (bits)')
legend(h6, 'Target Class 0','Target Class 1','Target Class 2','Target Class 3','Target Class 4','Target Class 5','Target Class 6','Target Class 7','Target Class 8','Target Class 9', 'Untargetted FGSM','Location','southeast','NumColumns',1);
export_fig(figure(26), 'TFGSM_IncorEnt_comparison.pdf');