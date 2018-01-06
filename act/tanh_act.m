function [f, df] = tanh_act(x)
f = tanh(x);
if nargout > 1
   df = 1 - f .^ 2; 
end