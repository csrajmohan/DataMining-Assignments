% [alllabels, allfeatures] = libsvmread('datasets\real-sim\real-sim');
% load savedvars\realsim_entire_data

% Separating training and test set
% cvFolds = crossvalind('Kfold', alllabels, 5);
% testIdx = (cvFolds == 1);
% trainIdx = ~testIdx;
% 
% trainfeatures = allfeatures(trainIdx == 1,:);
% testfeatures = allfeatures(testIdx == 1,:);
% 
% trainlabels = alllabels(trainIdx == 1,:);
% testlabels = alllabels(testIdx == 1,:);
% 
% clear alllabels allfeatures;

n = size(trainfeatures,1); % no of samples
d = size(trainfeatures,2); % dimension of each sample

no_of_iter = 50;
w = zeros(d,1);
realsim_perceptron_w = zeros(d,no_of_iter);
mistakes_per = zeros(no_of_iter,1);
flag = 1;
epoch = 0;

s = rng; % save seed

% randomly permute input samples
rng(s);
perm = randperm(n);

trainfeatures_transpose = trainfeatures';

while(flag == 1)
    
    flag = 0;
    epoch = epoch + 1;
    
    for j = 1: n
        
        i = perm(j);
        actual_label = trainlabels(i,1);
        vect_x = trainfeatures_transpose(:,i); 
        
        wt_x = w' * vect_x;
        
        if wt_x >= 0 
            yicap = 1;
        else
            yicap = -1;
        end
        
        if (actual_label * yicap) < 0        % if error
            w = w + (actual_label * vect_x);
            flag = 1;
            mistakes_per(epoch) = mistakes_per(epoch) + 1;
        end
        
    end
    
    str = ['epoch: ',num2str(epoch), ' mistakes: ', num2str(mistakes_per(epoch))];
    disp(str);
    
    realsim_perceptron_w(:,epoch) = w;
    if(epoch == no_of_iter)
        break;
    end
end
