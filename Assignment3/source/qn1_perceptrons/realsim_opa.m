% Realsim - OPA
% load savedvars\realsim_train_data
n = size(trainfeatures,1); % no of samples
d = size(trainfeatures,2); % dimension of each sample
w = rand(d,1); % 
opa_w = zeros(d,50);

flag = 1;
epoch = 0;

% save seed. do it only once for before running 3 algos
% randomly permute input samples

rng(s);
perm = randperm(n);
mistakes_opa = zeros(50,1);
trainfeatures_transpose = trainfeatures';

while(flag == 1)
   
    flag = 0;
    epoch = epoch + 1;
    
    for j = 1: n
        
        i = perm(j);
        
        actual_label = trainlabels(i,1);
        vect_x = trainfeatures_transpose(:,i);
        wt_x = w' * vect_x;
        
        % find sign of wt . x    
        if wt_x >= 0 
            yicap = 1;
        else
            yicap = -1;
        end
        
        % calculating tow
        m_yi_wt_xi = 1 - (actual_label * wt_x);
        norm_xisqr = (sum(vect_x .^ 2));
        
        if(norm_xisqr ~= 0)
            tow = max(0,m_yi_wt_xi) / norm_xisqr;
        else 
            tow = 0;
        end
           
        % updating weight vector
        if (actual_label * yicap) < 0        % if error
            w = w + ((tow * actual_label) * trainfeatures_transpose(:,i));
            flag = 1;         
            mistakes_opa(epoch) = mistakes_opa(epoch) + 1;     
        end
    end
    
    opa_w(:,epoch) = w;
    str = ['epoch: ',num2str(epoch), ' mistakes: ', num2str(mistakes_opa(epoch))];
    disp(str);
    if(epoch == 50)
        break;
    end
end
