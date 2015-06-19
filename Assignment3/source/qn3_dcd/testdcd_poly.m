
% clear training related variables for freeing memory

gamma = (1/22);
% load test data
[testlabels, testfeatures] = libsvmread('datasets\ijcnn\ijcnn1.t\ijcnn1.t');

Kez = (single(gamma * (testfeatures * Z'))) .^ 2;

ycap = sign(Kez * (M * w));

ycap = real(ycap);

idx_actual = find(testlabels == 1);
idx_predicted = find(ycap == 1);

accuracy = (sum(ycap == testlabels)) / size(testlabels,1) * 100;
testerr = 100 - accuracy;
disp(testerr);
