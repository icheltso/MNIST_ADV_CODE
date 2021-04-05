close all;
load('test.mat');
labels = data_test(:,1);
y = zeros(10,10000);
for i = 1:10000
    y(labels(i)+1,i) = 1;
end

images = data_test(:,2:785);
images = images/255;

temp_images = images;

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

batch4fgsmSSE = zeros(784,10);
batch4fgsmCE = zeros(784,10);

eps_steps = linspace(0,0.255,256);




maxdocval = zeros(1,256);

maxselarr = [1,10,50];

acc4batch = zeros(length(maxselarr),256);
ent4batch = zeros(length(maxselarr),256);
entcor4batch = zeros(length(maxselarr),256);
entincor4batch = zeros(length(maxselarr),256);
acc4batchCE = zeros(length(maxselarr),256);
ent4batchCE = zeros(length(maxselarr),256);
entcor4batchCE = zeros(length(maxselarr),256);
entincor4batchCE = zeros(length(maxselarr),256);

for msa = 1:length(maxselarr)
    max_selected = maxselarr(msa);
    DoCSSE = zeros(1,256);
    DoCCE = zeros(1,256);
    DocRand = zeros(1,256);
    acc_array = zeros(1,256);
    acc_array_CE = zeros(1,256);
    conf_mat_orig = zeros(10,10);
    conf_mat_128 = zeros(10,10);
    conf_mat_256 = zeros(10,10);
    conf_mat_128_CE = zeros(10,10);
    conf_mat_256_CE = zeros(10,10);

    entropies = zeros(1,256);
    entropiesRandSSE = zeros(1,256);
    entropiesRandCE = zeros(1,256);
    entropies_mat_128 = zeros(10,10);
    entropies_mat_256 = zeros(10,10);
    entropy_counts_128 = zeros(10,10);
    entropy_counts_256 = zeros(10,10);
    entropiesCE = zeros(1,256);
    entropies_mat_128CE = zeros(10,10);
    entropies_mat_256CE = zeros(10,10);
    entropy_counts_128CE = zeros(10,10);
    entropy_counts_256CE = zeros(10,10);



    ent_correct_guess_SSE = zeros(1,256);
    ent_incorrect_guess_SSE = zeros(1,256);
    ent_correct_guess_CE = zeros(1,256);
    ent_incorrect_guess_CE = zeros(1,256);
    
    for i = 0:9
        selected = 0;
        meangradSSE = zeros(784,1);
        meangradCE = zeros(784,1);
        [temp_images2,temp_labels] = rand_sample_selector(temp_images,labels,10000);
        for j = 1:10000
            if (selected ~= max_selected) && (temp_labels(j) == i)
                selected = selected + 1;
                meangradSSE = meangradSSE + CostGradient(w2,w3,w4,b2,b3,b4,transpose(temp_images(j,:)),label_to_vector(temp_labels(j)));
                meangradCE = meangradCE + CostGradientCE(w2CE,w3CE,w4CE,b2CE,b3CE,b4CE,transpose(temp_images(j,:)),label_to_vector(temp_labels(j)));
            end
        end
        batch4fgsmSSE(:,i+1) = meangradSSE/max_selected;
        batch4fgsmCE(:,i+1) = meangradCE/max_selected;
    end

for epsn = 1:256
    counter_bin = 0;
    eps_step = eps_steps(epsn);
    success = 0;
    successCE = 0;
for i = 1:n
    image_fgs = images(:,i) + eps_step * sign(batch4fgsmSSE(:,labels(i)+1));
    image_fgs = between01(image_fgs);
    image_fgs_CE = images(:,i) + eps_step * sign(batch4fgsmCE(:,labels(i)+1));
    image_fgs_CE = between01(image_fgs_CE);
    DoCSSE(1,epsn) = DoCSSE(1,epsn) + DoC(images(:,i),image_fgs);
    DoCCE(1,epsn) = DoCCE(1,epsn) + DoC(images(:,i),image_fgs_CE);
    out = NeuralF(w2,w3,w4,b2,b3,b4,image_fgs);
    outCE = NeuralF(w2CE,w3CE,w4CE,b2CE,b3CE,b4CE,image_fgs_CE);
    entropies(1,epsn) = entropies(1,epsn) + inf_entropy(out);
    entropiesCE(1,epsn) = entropiesCE(1,epsn) + inf_entropy(outCE);
    maxdocval(1,epsn) = maxdocval(1,epsn) + 28*eps_step/norm(images(:,i));
    big = 0;
    num = 0;
    bigCE = 0;
    numCE = 0;
    
for k = 1:10    %Choose most probable output
    if out(k) > big
        num = k-1;
        big = out(k);
    end
    if outCE(k) > bigCE
        numCE = k-1;
        bigCE = outCE(k);
    end
end


if labels(i) == num
    success = success + 1;
    ent_correct_guess_SSE(1,epsn) = ent_correct_guess_SSE(1,epsn) + inf_entropy(out);
else
    ent_incorrect_guess_SSE(1,epsn) = ent_incorrect_guess_SSE(1,epsn) + inf_entropy(out);
end

if labels(i) == numCE
    successCE = successCE + 1;
    ent_correct_guess_CE(1,epsn) = ent_correct_guess_CE(1,epsn) + inf_entropy(outCE);
else
    ent_incorrect_guess_CE(1,epsn) = ent_incorrect_guess_CE(1,epsn) + inf_entropy(outCE);
end

if epsn == 1 
conf_mat_orig(labels(i)+1,num+1) = conf_mat_orig(labels(i)+1,num+1) + 1;
end

if epsn == 128 
conf_mat_128(labels(i)+1,num+1) = conf_mat_128(labels(i)+1,num+1) + 1;
conf_mat_128_CE(labels(i)+1,numCE+1) = conf_mat_128_CE(labels(i)+1,numCE+1) + 1;
entropies_mat_128(labels(i)+1,num+1) = entropies_mat_128(labels(i)+1,num+1) + inf_entropy(out);
entropy_counts_128(labels(i)+1,num+1) = entropy_counts_128(labels(i)+1,num+1) + 1;
entropies_mat_128CE(labels(i)+1,numCE+1) = entropies_mat_128CE(labels(i)+1,numCE+1) + inf_entropy(outCE);
entropy_counts_128CE(labels(i)+1,numCE+1) = entropy_counts_128CE(labels(i)+1,numCE+1) + 1;
end

if epsn == 256 
conf_mat_256(labels(i)+1,num+1) = conf_mat_256(labels(i)+1,num+1) + 1;
conf_mat_256_CE(labels(i)+1,numCE+1) = conf_mat_256_CE(labels(i)+1,numCE+1) + 1;
entropies_mat_256(labels(i)+1,num+1) = entropies_mat_256(labels(i)+1,num+1) + inf_entropy(out);
entropy_counts_256(labels(i)+1,num+1) = entropy_counts_256(labels(i)+1,num+1) + 1;
entropies_mat_256CE(labels(i)+1,numCE+1) = entropies_mat_256CE(labels(i)+1,numCE+1) + inf_entropy(outCE);
entropy_counts_256CE(labels(i)+1,numCE+1) = entropy_counts_256CE(labels(i)+1,numCE+1) + 1;
end
    

end
maxdocval(1,epsn) = maxdocval(1,epsn)/n;
entropies(1,epsn) = entropies(1,epsn)/n;
entropiesCE(1,epsn) = entropiesCE(1,epsn)/n;
DoCSSE(1,epsn) = DoCSSE(1,epsn)/n;
DoCCE(1,epsn) = DoCCE(1,epsn)/n;
ent_correct_guess_SSE(1,epsn) = ent_correct_guess_SSE(1,epsn)/success;
ent_incorrect_guess_SSE(1,epsn) = ent_incorrect_guess_SSE(1,epsn)/(n-success);
ent_correct_guess_CE(1,epsn) = ent_correct_guess_CE(1,epsn)/successCE;
ent_incorrect_guess_CE(1,epsn) = ent_incorrect_guess_CE(1,epsn)/(n-successCE);

if epsn == 128
    entropies128 = entropies_mat_128./entropy_counts_128;
    entropies128CE = entropies_mat_128CE./entropy_counts_128CE;
end
if epsn == 256
    entropies256 = entropies_mat_256./entropy_counts_256;
    entropies256CE = entropies_mat_256CE./entropy_counts_256CE;
end

acc_array(epsn) = success/n*100;
acc_array_CE(epsn) = successCE/n*100;
fprintf('Stepsize: %f\n',eps_step);
fprintf('Accuracy SSE: ');
fprintf('%f',acc_array(epsn));
disp(' %');
fprintf('Accuracy CE: ');
fprintf('%f',acc_array_CE(epsn));
disp(' %');
fprintf('Entropy SSE: ');
fprintf('%f',entropies(1,epsn));
disp(' bits');
fprintf('Entropy CE: ');
fprintf('%f',entropiesCE(1,epsn));
disp(' bits');
fprintf('DoC SSE: ');
fprintf('%f\n',DoCSSE(1,epsn));
fprintf('DoC CE: ');
fprintf('%f\n',DoCCE(1,epsn));
end
acc4batch(msa,:) = acc_array;
acc4batchCE(msa,:) = acc_array_CE;
ent4batch(msa,:) = entropies;
ent4batchCE(msa,:) = entropiesCE;
entcor4batch(msa,:) = ent_correct_guess_SSE;
entincor4batch(msa,:) = ent_incorrect_guess_SSE;
entcor4batchCE(msa,:) = ent_correct_guess_CE;
entincor4batchCE(msa,:) = ent_incorrect_guess_CE;
end

for j = 1:10

conf_mat_orig(j,:) = round(conf_mat_orig(j,:)/each_label(j),3);
conf_mat_128(j,:) = round(conf_mat_128(j,:)/each_label(j),3);
conf_mat_256(j,:) = round(conf_mat_256(j,:)/each_label(j),3);
conf_mat_128_CE(j,:) = round(conf_mat_128_CE(j,:)/each_label(j),3);
conf_mat_256_CE(j,:) = round(conf_mat_256_CE(j,:)/each_label(j),3);
entropies128(j,:) = round(entropies128(j,:),3);
entropies256(j,:) = round(entropies256(j,:),3);
entropies128CE(j,:) = round(entropies128CE(j,:),3);
entropies256CE(j,:) = round(entropies256CE(j,:),3);

end


figure(1)
hold on
for k = 1:length(maxselarr)
plot(eps_steps,acc4batch(k,:),'DisplayName',sprintf('SSE, Batch Size: %d', maxselarr(k)));
plot(eps_steps,acc4batchCE(k,:),'DisplayName',sprintf('CE, Batch Size: %d', maxselarr(k)));
end
hold off
xlabel('\epsilon') 
ylabel('Accuracy (%)')
legend show
export_fig(figure(1), 'pert_acc_BFGSM_SSEvsCE.pdf')

figure(2)
xvals = {'0','1','2','3','4','5','6','7','8','9'};
yvals = {'0','1','2','3','4','5','6','7','8','9'};
conf_table1 = heatmap(xvals,yvals,conf_mat_orig);
conf_table1.Title = 'Original Confusion Matrix';
conf_table1.XLabel = 'Predicted Values';
conf_table1.YLabel = 'True Values';
export_fig(figure(2), 'BFGSM_orig_conf.pdf')

figure(3)
xvals = {'0','1','2','3','4','5','6','7','8','9'};
yvals = {'0','1','2','3','4','5','6','7','8','9'};
conf_table2 = heatmap(xvals,yvals,conf_mat_128);
conf_table2.Title = 'SSE BFGSM Confusion Matrix: \epsilon = 0.128';
conf_table2.XLabel = 'Predicted Values';
conf_table2.YLabel = 'True Values';
export_fig(figure(3), 'BFGSM_128_conf.pdf')

figure(4)
xvals = {'0','1','2','3','4','5','6','7','8','9'};
yvals = {'0','1','2','3','4','5','6','7','8','9'};
conf_table3 = heatmap(xvals,yvals,conf_mat_256);
conf_table3.Title = 'SSE BFGSM Confusion Matrix: \epsilon = 0.255';
conf_table3.XLabel = 'Predicted Values';
conf_table3.YLabel = 'True Values';
export_fig(figure(4), 'BFGSM_256_conf.pdf')

figure(5)
hold on
for k = 1:length(maxselarr)
plot(eps_steps,ent4batch(k,:),'DisplayName',sprintf('SSE, Batch Size: %d', maxselarr(k)));
plot(eps_steps,ent4batchCE(k,:),'DisplayName',sprintf('CE, Batch Size: %d', maxselarr(k)));
end
hold off
xlabel('\epsilon') 
ylabel('Average Entropy(bits)')
legend('location', 'southeast')
legend show
export_fig(figure(5), 'entropy_unt_BFGSM_curve.pdf');

figure(6)
xvals = {'0','1','2','3','4','5','6','7','8','9'};
yvals = {'0','1','2','3','4','5','6','7','8','9'};
conf_table1 = heatmap(xvals,yvals,entropies128);
conf_table1.Title = 'Per-Class Entropy - SSE for BFGSM Set, \epsilon = 0.128';
conf_table1.XLabel = 'Predicted Values';
conf_table1.YLabel = 'True Values';
export_fig(figure(6), 'entropy_conf_BFGSM128.pdf')

figure(7)
xvals = {'0','1','2','3','4','5','6','7','8','9'};
yvals = {'0','1','2','3','4','5','6','7','8','9'};
conf_table1 = heatmap(xvals,yvals,entropies256);
conf_table1.Title = 'Per-Class Entropy - SSE for BFGSM Set, \epsilon = 0.255';
conf_table1.XLabel = 'Predicted Values';
conf_table1.YLabel = 'True Values';
export_fig(figure(7), 'entropy_conf_BFGSM256.pdf')

figure(8)
xvals = {'0','1','2','3','4','5','6','7','8','9'};
yvals = {'0','1','2','3','4','5','6','7','8','9'};
conf_table1 = heatmap(xvals,yvals,entropies128CE);
conf_table1.Title = 'Per-Class Entropy - CE for BFGSM Set, \epsilon = 0.128';
conf_table1.XLabel = 'Predicted Values';
conf_table1.YLabel = 'True Values';
export_fig(figure(8), 'entropy_conf_BFGSM128CE.pdf')

figure(9)
xvals = {'0','1','2','3','4','5','6','7','8','9'};
yvals = {'0','1','2','3','4','5','6','7','8','9'};
conf_table1 = heatmap(xvals,yvals,entropies256CE);
conf_table1.Title = 'Per-Class Entropy - CE for BFGSM Set, \epsilon = 0.255';
conf_table1.XLabel = 'Predicted Values';
conf_table1.YLabel = 'True Values';
export_fig(figure(9), 'entropy_conf_BFGSM256CE.pdf')

figure(10)
xvals = {'0','1','2','3','4','5','6','7','8','9'};
yvals = {'0','1','2','3','4','5','6','7','8','9'};
conf_table2 = heatmap(xvals,yvals,conf_mat_128_CE);
conf_table2.Title = 'CE BFGSM Confusion Matrix: \epsilon = 0.128';
conf_table2.XLabel = 'Predicted Values';
conf_table2.YLabel = 'True Values';
export_fig(figure(10), 'BFGSM_128CE_conf.pdf')

figure(14)
xvals = {'0','1','2','3','4','5','6','7','8','9'};
yvals = {'0','1','2','3','4','5','6','7','8','9'};
conf_table3 = heatmap(xvals,yvals,conf_mat_256_CE);
conf_table3.Title = 'CE BFGSM Confusion Matrix: \epsilon = 0.255';
conf_table3.XLabel = 'Predicted Values';
conf_table3.YLabel = 'True Values';
export_fig(figure(14), 'BFGSM_256CE_conf.pdf')

figure(11)
hold on
h3(1)=plot(eps_steps,DoCSSE);
h3(2)=plot(eps_steps,DoCCE);
h3(3)=plot(eps_steps,maxdocval);
hold off
xlabel('\epsilon') 
ylabel('Degree of Change')
legend(h3,'DoC SSE','DoC CE', 'Max DoC')
export_fig(figure(11), 'pert_doc_BFGSM_SSEvsCE.pdf')

figure(12)
hold on
for k = 1:length(maxselarr)
plot(eps_steps,entcor4batch(k,:),'DisplayName',sprintf('SSE, Batch Size: %d', maxselarr(k)));
plot(eps_steps,entcor4batchCE(k,:),'DisplayName',sprintf('CE, Batch Size: %d', maxselarr(k)));
end
hold off
xlabel('\epsilon') 
ylabel('Average Entropy of Correct Guess (bits)')
legend('location', 'southeast')
legend show
export_fig(figure(12), 'cor_entropy_unt_BFGSM_curve.pdf');

figure(13)
hold on
for k = 1:length(maxselarr)
plot(eps_steps,entincor4batch(k,:),'DisplayName',sprintf('SSE, Batch Size: %d', maxselarr(k)));
plot(eps_steps,entincor4batchCE(k,:),'DisplayName',sprintf('CE, Batch Size: %d', maxselarr(k)));
end
hold off
xlabel('\epsilon') 
ylabel('Average Entropy of Incorrect Guess (bits)')
legend('location', 'southeast')
legend show
export_fig(figure(13), 'incor_entropy_unt_BFGSM_curve.pdf');

figure(15)
hold on
for k = 1:length(maxselarr)
plot(eps_steps,gradient(acc4batch(k,:)),'DisplayName',sprintf('SSE, Batch Size: %d', maxselarr(k)));
plot(eps_steps,gradient(acc4batchCE(k,:)),'DisplayName',sprintf('CE, Batch Size: %d', maxselarr(k)));
end
hold off
xlabel('\epsilon') 
ylabel('Numerical Gradient of Accuracy curves')
legend('location', 'southeast')
legend show
export_fig(figure(15), 'numgrad_acc_BFGSM_curve.pdf');
