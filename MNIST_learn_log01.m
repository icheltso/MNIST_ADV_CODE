load('train.mat');
labels = data(:,1);
imgs = data(:,2:785);
three_and_seven = find(labels == 3 | labels == 7,60000);
sz37 = size(three_and_seven,1);
bn37 = zeros(sz37,1);
img37 = zeros(sz37,784);
for k = 1:sz37
    img37(k,:) = imgs(three_and_seven(k),:);
    labels37(k,1) = labels(three_and_seven(k),1);
    if (labels37(k,1) == 7)
        bn37(k) = 1;
    end
end

%Now we make our images stored as column vectors.
img37_col = transpose(img37);
iter = 15000;
step = 0.00001;


%The next few lines randomly select N pictures and their respective labels.
N=1000; % no. of rows needed
[B,Blab] = rand_sample_selector(img37,bn37,N);
B_col = transpose(B);

array_to_img(B);

[B2,Blab2] = rand_sample_selector(imgs,labels,25);
