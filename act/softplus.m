function [f, df] = softplus(x)
f = log(1 + exp(x));
if nargout > 1
   df = 1 ./ (1 + exp(-x)); 
end