load('test.mat');
labels = data_test(:,1);
y = zeros(10,10000);
for i = 1:10000
    y(labels(i)+1,i) = 1;
end

images = data_test(:,2:785);
images = images/255;

images = images';

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
success = 0;
n = 10000;
correct_guess = zeros(1,784);
correct_labels = 0;
total_ent = 0;

entropies = zeros(10,10);
entropy_counts = zeros(10,10);

for i = 1:n
fprintf('Started iteration %f\n', i);
out2 = activate(images(:,i),w2,b2);
out3 = activate(out2,w3,b3);
out = activate(out3,w4,b4);
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
    correct_guess = cat(1,correct_guess,transpose(images(:,i)));
    correct_labels = cat(1,correct_labels,num);
end
total_ent = total_ent + inf_entropy(out);

entropies(labels(i)+1,num+1) = entropies(labels(i)+1,num+1) + inf_entropy(out);
entropy_counts(labels(i)+1,num+1) = entropy_counts(labels(i)+1,num+1) + 1;

end
ent_correct_guess = trace(entropies)/trace(entropy_counts);
ent_incorrect_guess = (sum(sum(entropies)) - trace(entropies))/(sum(sum(entropy_counts)) - trace(entropy_counts));
per_class_ent = sum(entropies,2);  %column vec
per_pred_ent = sum(entropies,1);   %row vec
entropies2 = entropies./entropy_counts;
per_class_ent = per_class_ent./sum(entropy_counts,2);
per_pred_ent = per_pred_ent./sum(entropy_counts,1);

total_ent = total_ent/n;

correct_guess = correct_guess(2:end,:);
correct_labels = correct_labels(2:end,:);
save('correct_guess.mat','correct_guess');
save('correct_labels.mat','correct_labels');
fprintf('Accuracy: ');
fprintf('%f',success/n*100);
disp(' %');
fprintf('Average Entropy: ');
fprintf('%f',total_ent);
disp(' bits');
fprintf('Average Entropy for correct predictions: ');
fprintf('%f',ent_correct_guess);
disp(' bits');
fprintf('Average Entropy for incorrect predictions: ');
fprintf('%f',ent_incorrect_guess);
disp(' bits');

for j = 1:10
entropies2(j,:) = round(entropies2(j,:),3);
end

figure(1)
yvals = {'True Class'};
xvals = {'0','1','2','3','4','5','6','7','8','9'};
conf_table1 = heatmap(xvals,yvals,transpose(per_class_ent));
conf_table1.Title = 'Per-Class Entropy - SSE for Original Set';
conf_table1.XLabel = 'Predicted Values';
conf_table1.YLabel = 'True Values';
export_fig(figure(1), 'per_cl_entropy_SSE.pdf')

figure(2)
yvals = {'Predicted Class'};
xvals = {'0','1','2','3','4','5','6','7','8','9'};
conf_table1 = heatmap(xvals,yvals,per_pred_ent);
conf_table1.Title = 'Per-Prediction Entropy - SSE for Original Set';
conf_table1.XLabel = 'Predicted Values';
conf_table1.YLabel = 'True Values';
export_fig(figure(2), 'per_pred_entropy_SSE.pdf')

figure(3)
xvals = {'0','1','2','3','4','5','6','7','8','9'};
yvals = {'0','1','2','3','4','5','6','7','8','9'};
conf_table1 = heatmap(xvals,yvals,entropies2);
conf_table1.Title = 'Per-Class Entropy - SSE for Original Set';
conf_table1.XLabel = 'Predicted Values';
conf_table1.YLabel = 'True Values';
export_fig(figure(3), 'entropy_conf.pdf')