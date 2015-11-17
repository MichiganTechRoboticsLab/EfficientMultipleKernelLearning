function K = svkernel( type, X, Y, gamma )

if( strcmp( type, 'linear' ) )
    
    K = X * Y';
    
elseif( strcmp( type, 'rbf' ) )
    
    nsqx = sum(X.^2, 2);
    nsqy = sum(Y.^2, 2);
    K = bsxfun(@minus, nsqx, (2*X)*Y.');
    K = bsxfun(@plus, nsqy.', K);
    K = exp(-gamma * K);
    
end