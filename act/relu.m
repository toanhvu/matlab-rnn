function [f, df] = relu(x)
f = max(0,x);
if nargout > 1
    df = double(x>0);
end