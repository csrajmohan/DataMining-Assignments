
n = size(trainfeatures,1); % no of samples
d = size(trainfeatures,2); % dimension of each sample
no_of_iter = 50;
w = rand(d,1);  % w ~= 0

realsim_mira_w = zeros(d,no_of_iter);
mistakes_mira = zeros(no_of_iter,1);

flag = 1;
epoch = 0;
tow = 0;

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
	   tow = 0;
        
        % find sign of wt . x
        if wt_x >= 0 
            yicap = 1;
        else
            yicap = -1;
        end
        
        % calculating tow
        yi_wt_xi = actual_label * wt_x;
       
        if(yi_wt_xi >= 0)
            tow = 0;
        else
            norm_xisqr = (sum(vect_x .^ 2));  %  dot(w, trainfeatures(1,:)) ^2  or  sumsqr(vect_x)
            if(norm_xisqr ~= 0)
                temp = yi_wt_xi/norm_xisqr;
            else 
                temp = 0;
            end
            if(temp <= -1)
                tow = 1;
            else         
                tow = -temp;
            end
        end
        
         
        % updating weight vector
        if (actual_label * yicap) < 0        % if error
            w = w + ((tow * actual_label) * vect_x);
            flag = 1;         
            mistakes_mira(epoch) = mistakes_mira(epoch) + 1;
        end
        
    end
    
    str = ['epoch: ',num2str(epoch), ' mistakes: ', num2str(mistakes_mira(epoch))];
    disp(str);
    
    realsim_mira_w(:,epoch) = w;
    if(epoch == no_of_iter)
        break;
    end
end
