function [cost, grad] = RNNcost(theta, input, output, hidden_size, lambda, gc)
%
input_size = size(input{1},1);
no_repeat = length(input);
output_size = size(output, 1);
batch_size = size(output, 2);

% params
limit0 = 1;
limit1 = hidden_size * input_size;
limit2 = limit1 + hidden_size * hidden_size;
limit3 = limit2 + output_size * hidden_size;
limit4 = limit3 + hidden_size;
limit5 = limit4 + output_size;

W1 = reshape(theta(limit0:limit1), [hidden_size, input_size]);
W2 = reshape(theta((limit1 +1):limit2), [hidden_size, hidden_size]);
W3 = reshape(theta((limit2 +1):limit3), [output_size, hidden_size]);
Bi = reshape(theta((limit3 +1):limit4), [hidden_size, 1]);
Bo = reshape(theta((limit4 +1):limit5), [output_size, 1]);

% forward pass

for i = 1:no_repeat
    if i==1
        Hp = repmat(zeros(hidden_size, 1), 1, batch_size);
    else
        Hp = H{i-1};
    end
    H_z{i} = W1 * input{i} + W2 * Hp + Bi * ones(1, batch_size);
    [H{i}, dH{i}] = tanh_act(H_z{i});
    clear H_z{i};    
end

O_z = W3 * H{no_repeat} + Bo * ones(1, batch_size);
O_z = O_z - repmat(max(O_z, [], 1), output_size,1);
O = exp(O_z) ./ repmat(sum(exp(O_z)), output_size,1);

loss = -1 * sum(sum(output .* log(O))) / batch_size;
weight_decay = 0.5 * lambda * (sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2)) + sum(sum(W3 .^ 2)));

cost = loss + weight_decay;

% backward pass
if nargout > 1
    W1_grad = zeros(hidden_size, input_size);
    W2_grad = zeros(hidden_size, hidden_size);
    Bi_grad = zeros(hidden_size, 1);
    
    del_O = -1 * (output - O); 
    W3_grad = del_O * transpose(H{no_repeat}); 
    Bo_grad = sum(del_O, 2);
            
    for i = no_repeat:-1:1
        if i==1
            Hp = repmat(zeros(hidden_size, 1), 1, batch_size);
        else
            Hp = H{i-1};
        end
        
        if i == no_repeat
            del_H{i} = (W3' * del_O) .* dH{i};
        else
            del_H{i} = (W2' * del_H{i+1}) .* dH{i};
        end
        W1_grad = W1_grad + del_H{i} * transpose(input{i});
        W2_grad = W2_grad + del_H{i} * transpose(Hp);
        Bi_grad = Bi_grad + sum(del_H{i}, 2);    
    end
    
    W1_grad = W1_grad / batch_size;    
    W2_grad = W2_grad / batch_size; 
    W3_grad = W3_grad / batch_size;
    Bi_grad = Bi_grad / batch_size;
    Bo_grad = Bo_grad / batch_size;
    
    % gradient clipping
    W1_grad = w_clip(W1_grad, W1, lambda, gc);
    W2_grad = w_clip(W2_grad, W2, lambda, gc);
    W3_grad = w_clip(W3_grad, W3, lambda, gc);
    Bi_grad = b_clip(Bi_grad, gc);
    Bo_grad = b_clip(Bo_grad, gc);
    
    grad = [W1_grad(:); W2_grad(:); W3_grad(:); Bi_grad(:); Bo_grad(:)];
end