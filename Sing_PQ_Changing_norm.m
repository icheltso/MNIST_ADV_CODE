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
niter = 40;
ranges = 100;
batch = 500;
faktor = 0.05;
p = Inf;
q = 10;
test_sz = 11;
pnorm = [1.2,2,3,4,5,6,7,8,9,10,Inf];
qnorm = [1,2,3,4,5,6,7,8,9,10,20];

changing_norm_acc = zeros(test_sz,test_sz);
changing_norm_sr = zeros(test_sz,test_sz);
changing_norm_ent = zeros(test_sz,test_sz);
changing_norm_time = zeros(test_sz,test_sz);

avg_niter = 10;   %Number of iterations to reduce discrepancies

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

[X,Xlab] = rand_sample_selector(images,labels,batch);
 X = X';

for m = 1:test_sz
for j = 1:test_sz
for ni = 1:avg_niter
        fprintf('Starting Case: (p,q) = (%f,%f), Iteration: %f\n', [pnorm(m),qnorm(j),ni]);
        tStart = tic;
        [v,mu] = Singular_PQ(w2,w3,w4,b2,b3,b4,X,niter,pnorm(m),qnorm(j));
        tEnd = toc(tStart);
        changing_norm_time(m,j) = changing_norm_time(m,j) + tEnd;
        success2 = 0;
        failure2 = 0;
        entropies = 0;
for i = 1:n
    image_attack = images_temp(:,i) + faktor * v * norm(images_temp(:,i));
    out2 = NeuralF(w2,w3,w4,b2,b3,b4,image_attack);
    entropies = entropies + inf_entropy(out2);
    
    
    big2 = 0;
    num2 = 0;
    
    for k = 1:10    
    
        if out2(k) > big2
            num2 = k-1;
            big2 = out2(k);
        end

    end
    
    
    if labels(i) == num2   %For finding accuracy of the Network for FGSM-set
        success2 = success2 + 1;
    elseif ((labels(i) ~= num2) && (labels(i) == num(i))) %For finding SR of FGSM-set
        failure2 = failure2 + 1;
    end
    
end
changing_norm_ent(m,j) = changing_norm_ent(m,j) + entropies/n;

changing_norm_sr(m,j) = changing_norm_sr(m,j) + failure2/success*100;

changing_norm_acc(m,j) = changing_norm_acc(m,j) + success2/n*100;

end
end
end
changing_norm_time = changing_norm_time/avg_niter;
changing_norm_ent = changing_norm_ent/avg_niter;
changing_norm_sr = changing_norm_sr/avg_niter;
changing_norm_acc = changing_norm_acc/avg_niter;

for j=1:test_sz
        changing_norm_ent(j,:) = round(changing_norm_ent(j,:),3);
        changing_norm_acc(j,:) = round(changing_norm_acc(j,:),3);
        changing_norm_sr(j,:) = round(changing_norm_sr(j,:),3);
        changing_norm_time(j,:) = round(changing_norm_time(j,:),3);
end
    
    xvals = {'1','2','3','4','5','6','7','8','9','10','20'};
    yvals = {'1.2','2','3','4','5','6','7','8','9','10','Inf'};


    figure(1)
    conf_table2 = heatmap(xvals,yvals,changing_norm_ent);
    conf_table2.Title = strcat('(p,q)-singular method entropies: Batch Size = 500, DoC = 0.05');
    conf_table2.XLabel = 'q';
    conf_table2.YLabel = 'p';
    export_fig(figure(1), sprintf('UNI_Singpq_enthist.pdf'));


    figure(2)
    conf_table2 = heatmap(xvals,yvals,changing_norm_acc);
    conf_table2.Title = strcat('(p,q)-singular method accuracy scores: Batch Size = 500, DoC = 0.05');
    conf_table2.XLabel = 'q';
    conf_table2.YLabel = 'p';
    export_fig(figure(2), sprintf('UNI_Singpq_acchist.pdf'));
    
    
    figure(3)
    conf_table2 = heatmap(xvals,yvals,changing_norm_sr);
    conf_table2.Title = strcat('(p,q)-singular method SR-scores: Batch Size = 500, DoC = 0.05');
    conf_table2.XLabel = 'q';
    conf_table2.YLabel = 'p';
    export_fig(figure(3), sprintf('UNI_Singpq_srhist.pdf'));

    figure(4)
    conf_table2 = heatmap(xvals,yvals,changing_norm_time);
    conf_table2.Title = strcat('(p,q)-singular method TTA: Batch Size = 500');
    conf_table2.XLabel = 'q';
    conf_table2.YLabel = 'p';
    export_fig(figure(4), sprintf('UNI_Singpq_timehist.pdf'));

