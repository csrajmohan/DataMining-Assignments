% Perceptron Learning Algorithm for Binary Classification

clear;
% read the dataset
[trainlabels, trainfeatures] = libsvmread('datasets\rcv\rcv1_train.binary');

n = size(trainfeatures,1); % no of samples
d = size(trainfeatures,2); % dimension of each sample
max_epoch = 50;
w = zeros(d,1);
perceptron_w = zeros(d,max_epoch);
mistakes_per = zeros(max_epoch,1);
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
    
    perceptron_w(:,epoch) = w;
    if(epoch == max_epoch)
       break;
    end
end    
