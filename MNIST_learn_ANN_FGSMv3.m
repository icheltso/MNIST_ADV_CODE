load('correct_guess.mat')
load('correct_labels.mat')

correct_guess = correct_guess';

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
n = 9199;

success = 0;
eps_step = 0.255;
for i = 1:n
    image_fgs = correct_guess(:,i) + eps_step * sign(CostGradient(w2,w3,w4,b2,b3,b4,correct_guess(:,i),label_to_vector(correct_labels(i))));
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


if correct_labels(i) == num
    success = success + 1;
end

end

acc = success/n*100;
fprintf('Stepsize: %f\n',eps_step);
fprintf('Accuracy: ');
fprintf('%f',acc);
disp(' %');




dummy_cat = dummy_cat(2:end,:);

figure(1)
array_to_img(dummy_cat);
export_fig(figure(1),'pert_three.pdf');
figure(2)
plot(eps_steps,acc_array);
xlabel('\epsilon') 
ylabel('Correctly identified perturbated originals (%)')
export_fig(figure(2), 'pert_accv2.pdf');
misclass_reshp = transpose(reshape(misclassed_three,[16,16]));
writematrix(misclass_reshp,'misclass_reshp.csv');