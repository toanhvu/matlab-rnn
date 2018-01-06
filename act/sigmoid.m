function [f, df] = sigmoid(x)
f = 1 ./ (1 + exp(-x));
if nargout > 1
   df = f .* (1 - f); 
end