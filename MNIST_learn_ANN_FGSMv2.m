load('correct_guess.mat')
load('correct_labels.mat')

correct_guess = correct_guess';

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
n = 9182;

eps_steps = linspace(0,0.255,256);
SR_array = zeros(1,256);
SR_array_CE = zeros(1,256);
dummy_cat = zeros(1,784);
misclassed_three = zeros(1,256);
entropies = zeros(1,256);
entropies_mat_128 = zeros(10,10);
entropies_mat_256 = zeros(10,10);
entropy_counts_128 = zeros(10,10);
entropy_counts_256 = zeros(10,10);
entropiesCE = zeros(1,256);
entropies_mat_128CE = zeros(10,10);
entropies_mat_256CE = zeros(10,10);
entropy_counts_128CE = zeros(10,10);
entropy_counts_256CE = zeros(10,10);

for epsn = 1:256
    eps_step = eps_steps(epsn);
    success = 0;
    successCE = 0;
for i = 1:n
    image_fgs = correct_guess(:,i) + eps_step * sign(CostGradient(w2,w3,w4,b2,b3,b4,correct_guess(:,i),label_to_vector(correct_labels(i))));
    image_fgs = between01(image_fgs);
    image_fgs_CE = correct_guess(:,i) + eps_step * sign(CostGradientCE(w2CE,w3CE,w4CE,b2CE,b3CE,b4CE,correct_guess(:,i),label_to_vector(correct_labels(i))));
    image_fgs_CE = between01(image_fgs_CE);
    out = NeuralF(w2,w3,w4,b2,b3,b4,image_fgs);
    outCE = NeuralF(w2CE,w3CE,w4CE,b2CE,b3CE,b4CE,image_fgs_CE);
    entropies(1,epsn) = entropies(1,epsn) + inf_entropy(out);
    entropiesCE(1,epsn) = entropiesCE(1,epsn) + inf_entropy(outCE);
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


if correct_labels(i) == num
    success = success + 1;
end

if correct_labels(i) == numCE
    successCE = successCE + 1;
end
   
if i == 62
    dummy_cat = cat(1,dummy_cat,transpose(image_fgs));
    misclassed_three(epsn) = num;
end

if epsn == 128
entropies_mat_128(correct_labels(i)+1,num+1) = entropies_mat_128(correct_labels(i)+1,num+1) + inf_entropy(out);
entropy_counts_128(correct_labels(i)+1,num+1) = entropy_counts_128(correct_labels(i)+1,num+1) + 1;
entropies_mat_128CE(correct_labels(i)+1,numCE+1) = entropies_mat_128CE(correct_labels(i)+1,numCE+1) + inf_entropy(outCE);
entropy_counts_128CE(correct_labels(i)+1,numCE+1) = entropy_counts_128CE(correct_labels(i)+1,numCE+1) + 1;
end

if epsn == 256
entropies_mat_256(correct_labels(i)+1,num+1) = entropies_mat_256(correct_labels(i)+1,num+1) + inf_entropy(out);
entropy_counts_256(correct_labels(i)+1,num+1) = entropy_counts_256(correct_labels(i)+1,num+1) + 1;
entropies_mat_256CE(correct_labels(i)+1,numCE+1) = entropies_mat_256CE(correct_labels(i)+1,numCE+1) + inf_entropy(outCE);
entropy_counts_256CE(correct_labels(i)+1,numCE+1) = entropy_counts_256CE(correct_labels(i)+1,numCE+1) + 1;
end

end
entropies(1,epsn) = entropies(1,epsn)/n;
entropiesCE(1,epsn) = entropiesCE(1,epsn)/n;

if epsn == 128
    entropies128 = entropies_mat_128./entropy_counts_128;
    entropies128CE = entropies_mat_128CE./entropy_counts_128CE;
end
if epsn == 256
    entropies256 = entropies_mat_256./entropy_counts_256;
    entropies256CE = entropies_mat_256CE./entropy_counts_256CE;
end


SR_array(epsn) = (n-success)/n*100;
SR_array_CE(epsn) = (n-successCE)/n*100;
fprintf('Stepsize: %f\n',eps_step);
fprintf('Accuracy: ');
fprintf('%f',SR_array(epsn));
disp(' %');
fprintf('Entropy SSE: ');
fprintf('%f',entropies(1,epsn));
disp(' bits');
fprintf('Entropy CE: ');
fprintf('%f',entropiesCE(1,epsn));
disp(' bits');
end





dummy_cat = dummy_cat(2:end,:);
extract_cat = dummy_cat([1,86,171,256],:);
figure(1)
array_to_img(dummy_cat);
export_fig(figure(1),'pert_three.pdf');

figure(2)
hold on
h1(1) = plot(eps_steps,SR_array);
h1(2) = plot(eps_steps,SR_array_CE);
hold off
xlabel('\epsilon') 
ylabel('Success rate (%)')
legend(h1,'SR SSE','SR CE')
export_fig(figure(2), 'pert_SR_SSEvsCE.pdf');

misclass_reshp = transpose(reshape(misclassed_three,[16,16]));
writematrix(misclass_reshp,'misclass_reshp.csv');
figure(3)
array_to_img(extract_cat);
export_fig(figure(3),'pert_three_extract.pdf');

bad62 = correct_guess(:,67) + 2*sign(CostGradient(w2,w3,w4,b2,b3,b4,correct_guess(:,67),label_to_vector(correct_labels(67))));
bad62 = bad62';
figure(4)
imshow(array_to_imgv2(bad62));
export_fig(figure(4),'pert_extreme_extract.pdf');


for j = 1:10
entropies128(j,:) = round(entropies128(j,:),3);
entropies256(j,:) = round(entropies256(j,:),3);
entropies128CE(j,:) = round(entropies128CE(j,:),3);
entropies256CE(j,:) = round(entropies256CE(j,:),3);
end