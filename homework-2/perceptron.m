X = [1 1 1 -1
     -1 1 1 -1
     1 1 -1 -1
     -1 1 -1 -1
     1 -1 1 1
     -1 -1 1 1
     1 -1 -1 1
     -1 -1 -1 1];
T = [-1 -1
     -1 -1
     -1 1
     -1 1
     1 -1
     1 -1
     1 1
     1 1];
W = [0 0
     0 0
     0 0
     0 0];
b = [0 0];
alpha = 1;
theta = 0.2;

size_X = size(X);

clc
iter = 17;
for j = 0:(iter - 1)
    i = mod(j, size_X(1)) + 1;
    
    x = X(i,:);
    t = T(i,:);
    
    % Feed-forward
    Yin = x*W + b;

    % Bipolar Heaviside unit step function
    Y(Yin >= theta) = 1;
    Y(Yin < theta) = -1;
    
    % Bipolar Heaviside unit step function with half-maximum convention
%     Y(Yin > theta) = 1;
%     Y(Yin == theta) = 0;
%     Y(Yin < -theta) = -1;
    
    % With uncertainty margins
%     Y(Yin > theta) = 1;
%     Y(abs(Yin) <= theta) = 0;
%     Y(Yin < -theta) = -1;

    % Delta rule
    errorW = x'*(Y - t);
    deltaW = 2*alpha*errorW;
    errorB = (Y - t);
    deltaB = 2*alpha*errorB;
    W = W - deltaW;
    b = b - deltaB;
    
        j+1
        x
        t
        Yin
        Y
        errorW
        errorB
        W
        b
end