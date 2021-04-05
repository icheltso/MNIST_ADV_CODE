load('test.mat');
labels = data_test(:,1);
imgs = data_test(:,2:785);

[B2,Blab2] = rand_sample_selector(imgs,labels,25);
img = array_to_imgv2(B2);

figure(1)
imshow(img);
export_fig(figure(1), 'draw_test25.pdf');
