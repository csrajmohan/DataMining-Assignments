% realsim-test.m

%[testlabels, testfeatures] = libsvmread('datasets\realsim\realsim_test.binary');
%load savedvars\realsim_perceptron_w.mat
%big_w=realsim_per_w;

n = size(testfeatures,1); % no of samples
d = size(testfeatures,2);% dimension of each sample

noof_iter = 50; % no of epochs of training(for each w check accuracy)
predictedlabels = int8(ones(n,1));
test_mistakes = zeros(noof_iter,1);
accuracy = zeros(noof_iter,1);

% w is fixed

testfeatures_transpose = testfeatures';    
clear testfeatures;
for iter = 1 : noof_iter
    w = big_w(:,iter);
    for i = 1: n
        actual_label = testlabels(i,1);
        vect_x = testfeatures_transpose(:,i);

        wt_x = w' * vect_x;

        if wt_x <0
            predictedlabels(i,1) = -1;
        else
            predictedlabels(i,1) = 1;
        end
    end
    
    accuracy(iter,1) = 100 - (sum(testlabels == predictedlabels)/n)*100;
    str = ['test error: ', num2str(accuracy(iter,1))];
    disp(str);
    
end
