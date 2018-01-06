function [cost, grad, n1, n5, n10] = SGRNN_cost(theta, input, output, hidden_size, lamda, gc)
theta1 = theta.theta1;
theta2 = theta.theta2;
theta0 = theta.theta0;

input_size = size(input{1}, 1);
output_size = size(output, 1);
batch_size = size(input{1}, 2);
no_loop = length(input);

%% params
% theta1
limit0 = 1;
limit1 = hidden_size * input_size;
limit2 = limit1 + hidden_size * hidden_size;
limit3 = limit2 + hidden_size;
limit4 = limit3 + hidden_size;

W1 = reshape(theta1(limit0:limit1), [hidden_size, input_size]);
W2 = reshape(theta1((limit1 +1):limit2), [hidden_size, hidden_size]);
Bk = reshape(theta1((limit2 +1):limit3), [hidden_size, 1]);
Bh = reshape(theta1((limit3 +1):limit4), [hidden_size, 1]);

% theta2
a = reshape(theta2(1:hidden_size), [hidden_size, 1]);
b1 = reshape(theta2((hidden_size+1):2*hidden_size), [hidden_size, 1]);
b2 = reshape(theta2((2*hidden_size +1):3*hidden_size), [hidden_size, 1]);

%theta0
limit0 = 1;
limit1 = output_size * hidden_size;
limit2 = limit1 + output_size;
Wo = reshape(theta0(limit0:limit1), [output_size, hidden_size]);
Bo = reshape(theta0((limit1 +1):limit2), [output_size, 1]);

%% forward 
for i=1:no_loop
    if i==1
        Hp = zeros(hidden_size, batch_size);
    else
        Hp = H{i-1};
    end
    K_z = W1*input{i} + W2*Hp + Bk * ones(1, batch_size);
    [K{i}, dK{i}] = tanh_act(K_z);
    
    H_z = repmat(a, [1, batch_size]).* Hp .* K{i} + repmat(b1, [1, batch_size]) .* Hp + repmat(b2, [1, batch_size]) .* K{i} + Bh * ones(1, batch_size);
    [H{i}, dH{i}] = linear_act(H_z);    
end
O_z = Wo * H{no_loop} + Bo * ones(1, batch_size);
O_z = O_z - repmat(max(O_z, [], 1), output_size,1);
O = exp(O_z) ./ repmat(sum(exp(O_z)), output_size,1);

loss = -1 * sum(sum(output .* log(O))) / batch_size;
weight_decay = 0.5 * lamda * (sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2)) + sum(sum(Wo .^ 2)));

cost = loss + weight_decay;

%% backward
if nargout>1
    W1_grad = zeros(hidden_size, input_size);
    W2_grad = zeros(hidden_size, hidden_size);
    Bk_grad = zeros(hidden_size, 1);
    Bh_grad = zeros(hidden_size, 1);

    a_grad = zeros(hidden_size, 1);
    b1_grad = zeros(hidden_size, 1);
    b2_grad = zeros(hidden_size, 1);

    % at time t = no_repeat
    del_O = -1 * (output - O); 
    Wo_grad = del_O * transpose(H{no_loop}); 
    Bo_grad = sum(del_O, 2);
    
    for i=no_loop:-1:1
        if i==1
            Hp = zeros(hidden_size, batch_size);
        else
            Hp = H{i-1};
        end
        if i == no_loop
            del_H{i} = (Wo' * del_O) .* dH{i};
        else
            del_H{i} = (W2' * del_K{i+1} + (repmat(a, [1, batch_size]) .* K{i+1} + repmat(b1, [1, batch_size])) .* del_H{i+1}) .* dH{i};
        end         
        
        Bh_grad = Bh_grad + sum(del_H{i}, 2);
        a_grad = a_grad + sum(Hp .* K{i} .* del_H{i}, 2);
        b1_grad = b1_grad + sum(Hp .* del_H{i}, 2);
        b2_grad = b2_grad + sum(K{i} .* del_H{i}, 2);
        
        del_K{i} = (repmat(a, [1, batch_size]) .* Hp + repmat(b2, [1, batch_size])).*del_H{i} .* dK{i};
        W1_grad = W1_grad + del_K{i} * transpose(input{i});
        W2_grad = W2_grad + del_K{i} * transpose(Hp);
        Bk_grad = Bk_grad + sum(del_K{i}, 2);
        
        if i==1
            n1 = mean(log10(sqrt(sum(del_H{i}.^2,1))));
        elseif i==5
            n5 = mean(log10(sqrt(sum(del_H{i}.^2,1))));
        elseif i==10
            n10 = mean(log10(sqrt(sum(del_H{i}.^2,1))));
        end
    end
    
    W1_grad = W1_grad / batch_size;
    W2_grad = W2_grad / batch_size;
    Bk_grad = Bk_grad / batch_size;
    Bh_grad = Bh_grad / batch_size;
    
    Wo_grad = Wo_grad / batch_size;
    Bo_grad = Bo_grad / batch_size;
    
    a_grad = a_grad / batch_size;
    b1_grad = b1_grad / batch_size;
    b2_grad = b2_grad / batch_size;
    
    % gradient clipping
    W1_grad = w_clip(W1_grad, W1, lamda, gc);
    W2_grad = w_clip(W2_grad, W2, lamda, gc);
    Bk_grad = b_clip(Bk_grad, gc);
    Bh_grad = b_clip(Bh_grad, gc);
    
    Wo_grad = w_clip(Wo_grad, Wo, lamda, gc);
    Bo_grad = b_clip(Bo_grad, gc);
    
    grad1 = [W1_grad(:); W2_grad(:); Bk_grad(:); Bh_grad(:)];
    grad0 = [Wo_grad(:); Bo_grad(:)];
    grad2 = [a_grad(:); b1_grad(:); b2_grad(:)];
    
    grad.grad0 = grad0;
    grad.grad1 = grad1;
    grad.grad2 = grad2;
end












