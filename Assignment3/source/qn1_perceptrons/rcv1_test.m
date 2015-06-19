
%load savedvars\perceptron_w
%[testlabels, testfeatures] = libsvmread('datasets\rcv\rcv1_test.binary');
% enable below line based on which algorithm you are going to test after
% loading weight vector obtained by training on data with that algorithm
%big_w=perceptron_w;
n = 677399; % no of samples
d = 47236; % dimension of each sample

predictedlabels = int8(ones(n,1));
test_mistakes = zeros(100,1);
accuracy = zeros(100,1);

% w is fixed

testfeatures_transpose = testfeatures';    

for iter = 1 : 50
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
    
    accuracy(iter,1) = (sum(testlabels == predictedlabels)/n)*100;
    %str = ['mistakes: ', num2str(test_mistakes(iter,1))];
    str = ['accuracy: ', num2str(accuracy(iter,1))];
    disp(str);
    
end


