% Nystrom approximation and Dual co-ordinate descent method
C = 10;      

p = 400;   % size(Centroid,1);  % Number of Samples
n = 49990;  % # of training data points
d = 22;     % Dimension of training data point

gamma = (1/22);

% load training data
[trainlabels, trainfeatures] = libsvmread('datasets\ijcnn\ijcnn1\ijcnn1');

Z = zeros(p,d);
w = zeros(p,1);
alpha = zeros(1,n);
flag = 1;
maxiter = 10;

% Pick p samples using k-means centroids
% opts = statset('MaxIter', 50, 'Display', 'iter');
% [idx, Centroid, sumd,D] = kmeans(trainfeatures, p, 'options',opts, 'EmptyAction','singleton', 'replicates',1);
%for i = 1 : p
%    Z(i,:) = Centroid(i,:);
%end

% Pick 'p' samples randomly
 p_pts = randperm(n,p);
 for i = 1 : p
     Z(i,:) = trainfeatures(p_pts(i),:);
 end

% let kernel function be linear : k(xi,xj) = xi.xj
Zt = Z';

% Find Kzz
Kzz = (single(gamma * (Z * Zt))) .^ 2;

% Eigen value decomposition
[EigVects, EigVals]  = eig(Kzz);  % Kzz = EigVects * EigVals * EigVects'

% Find M
M = EigVects * (EigVals ^ (-1/2));

% Find Krz
Krz = (single(gamma * (trainfeatures * Zt))) .^ 2;

% Find FrCap
FrCap = single(Krz * M);

% Train linear SVM using Dual Co-ordinate Descent(DCD) method

for iter = 1:maxiter
    
    mistks = 0;

    for i = 1:n

        xi = FrCap(i,:);
        xit = xi';
        yi = trainlabels(i);
        yiwtxi = yi * (xi * w);

        % If KKT conditions are not satisfied
        if( ~(((alpha(i) == 0) && (yiwtxi >= 1)) || ((alpha(i) == C) && (yiwtxi <= 1)) || ((alpha(i) > 0) && (alpha(i) < C) && (yiwtxi == 1))))
            mistks = mistks + 1;
            towcap = (1-yiwtxi) / (xi * xit);

            if(towcap <= -alpha(i))
                tow = -alpha(i);
            elseif(towcap >= C - alpha(i))
                tow = C-alpha(i);
            else
                tow = towcap;   
            end

            % Update weight vector and alpha with tow
            w = w + (tow * yi) * xit;
            alpha(i) = alpha(i) + tow;   
        end
    end
end
