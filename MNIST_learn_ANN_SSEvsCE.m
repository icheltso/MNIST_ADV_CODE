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
success2 = 0;
n = 10000;
correct_guess = zeros(1,784);
correct_labels = 0;
total_ent = 0;

entropies = zeros(10,10);
entropy_counts = zeros(10,10);
incorrect_pics = zeros(1,784);
counterinc = 0;

for i = 1:n
fprintf('Started iteration %f\n', i);
out = NeuralF(w2,w3,w4,b2,b3,b4,images(:,i));
outCE = NeuralF(w2CE,w3CE,w4CE,b2CE,b3CE,b4CE,images(:,i));
big = 0;
num = 0;
big2 = 0;
num2 = 0;
for k = 1:10    %Choose most probable output
    if out(k) > big
        num = k-1;
        big = out(k);
    end
end

for k = 1:10    %Choose most probable output
    if outCE(k) > big2
        num2 = k-1;
        big2 = outCE(k);
    end
end

if (labels(i) ~= num) && (labels(i) ~= num2) && (counterinc < 25)
    incorrect_pics = cat(1,incorrect_pics,transpose(images(:,i)));
    counterinc = counterinc + 1;
end

end

incorrect_pics(1,:) = [];

figure(1)
imshow(array_to_imgv2(incorrect_pics));
export_fig(figure(1), 'incorrect25SSEandCE.pdf')
