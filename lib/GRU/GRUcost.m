function [cost, grad] = GRUcost(theta, input, output, hidden_size, lamda, gc)
input_size = size(input{1}, 1);
no_repeat = length(input);
output_size = size(output, 1);
batch_size = size(output, 2);

% params
limit0 = 1;
limit1 = hidden_size * input_size;
limit2 = limit1 + hidden_size * hidden_size;
limit3 = limit2 + hidden_size * input_size;
limit4 = limit3 + hidden_size * hidden_size;
limit5 = limit4 + hidden_size * input_size;
limit6 = limit5 + hidden_size * hidden_size;
limit7 = limit6 + output_size * hidden_size;
limit8 = limit7 + hidden_size;
limit9 = limit8 + hidden_size;
limit10 = limit9 + hidden_size;
limit11 = limit10 + output_size;

W1 = reshape(theta(limit0:limit1), [hidden_size, input_size]);
W2 = reshape(theta((limit1+1):limit2), [hidden_size, hidden_size]);
U1 = reshape(theta((limit2+1):limit3), [hidden_size, input_size]);
U2 = reshape(theta((limit3+1):limit4), [hidden_size, hidden_size]);
V1 = reshape(theta((limit4+1):limit5), [hidden_size, input_size]);
V2 = reshape(theta((limit5+1):limit6), [hidden_size, hidden_size]);
W3 = reshape(theta((limit6+1):limit7), [output_size, hidden_size]);
Bk = reshape(theta((limit7+1):limit8), [hidden_size, 1]);
Bz = reshape(theta((limit8+1):limit9), [hidden_size, 1]);
Br = reshape(theta((limit9+1):limit10), [hidden_size, 1]);
Bo = reshape(theta((limit10+1):limit11), [output_size, 1]);

% forward pass
% at time t = 1
h0 = zeros(hidden_size, 1);
H0 = repmat(h0, 1, batch_size);
R_z{1} = V1 * input{1} + V2 * H0 + Br * ones(1, batch_size);
[R{1}, dR{1}] = sigmoid(R_z{1});
Z_z{1} = U1 * input{1} + U2 * H0 + Bz * ones(1, batch_size);
[Z{1}, dZ{1}] = sigmoid(Z_z{1});
K_z{1} = W1 * input{1} + R{1} .* (W2 * H0) + Bk * ones(1, batch_size);
[K{1}, dK{1}] = tanh_act(K_z{1});
H_z{1} = (1 - Z{1}) .* K{1};
[H{1}, dH{1}] = linear_act(H_z{1});

clear R_z{1} Z_z{1} K_z{1} H_z{1};

% in range time t = 1:no_repeat
for i=2:no_repeat
    R_z{i} = V1 * input{i} + V2 * H{i-1} + Br * ones(1, batch_size);
    [R{i}, dR{i}] = sigmoid(R_z{i});
    Z_z{i} = U1 * input{i} + U2 * H{i-1} + Bz * ones(1, batch_size);
    [Z{i}, dZ{i}] = sigmoid(Z_z{i});
    K_z{i} = W1 * input{i} + R{i} .* (W2 * H{i-1}) + Bk * ones(1, batch_size);
    [K{i}, dK{i}] = tanh_act(K_z{i});
    H_z{i} = Z{i} .* H{i-1} + (1-Z{i}) .* K{i};
    [H{i}, dH{i}] = linear_act(H_z{i});
    
    clear R_z{i} Z_z{i} K_z{i} H_z{i};    
end

O_z = W3 * H{no_repeat} + Bo * ones(1, batch_size);
O_z = O_z - repmat(max(O_z, [], 1), output_size,1);
O = exp(O_z) ./ repmat(sum(exp(O_z)), output_size,1);

loss = -1 * sum(sum(output .* log(O))) / batch_size;
weight_decay = 0.5 * lamda * (sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2)) + sum(sum(W3 .^ 2)) + sum(sum(U1 .^ 2)) + sum(sum(U2 .^ 2)) + sum(sum(V1 .^ 2)) + sum(sum(V2 .^ 2)));

cost = loss + weight_decay;

% backward pass
if nargout > 1
    W1_grad = zeros(hidden_size, input_size);
    W2_grad = zeros(hidden_size, hidden_size);
    V1_grad = zeros(hidden_size, input_size);
    V2_grad = zeros(hidden_size, hidden_size);
    U1_grad = zeros(hidden_size, input_size);
    U2_grad = zeros(hidden_size, hidden_size);
    Bk_grad = zeros(hidden_size, 1);
    Bz_grad = zeros(hidden_size, 1);
    Br_grad = zeros(hidden_size, 1);
    
    % at time t = no_repeat
    del_O = -1 * (output - O); 
    W3_grad = del_O * transpose(H{no_repeat}); 
    Bo_grad = sum(del_O, 2);
                
    % in range t = no_repeat:-1:2    
    for i = no_repeat:-1:2
        if i == no_repeat
            del_H{i} = (W3' * del_O) .* dH{i};
        else
            del_H{i} = (U2' * del_Z{i+1} + V2' * del_R{i+1} + Z{i+1} .* del_H{i+1} + W2' * (R{i+1} .* del_K{i+1})) .* dH{i};
        end
        
        del_K{i} = ((1-Z{i}) .* del_H{i}) .* dK{i};
        W1_grad = W1_grad + del_K{i} * transpose(input{i});
        W2_grad = W2_grad + (R{i} .* del_K{i}) * transpose(H{i-1});
        Bk_grad = Bk_grad + sum(del_K{i}, 2);
                      
        del_Z{i} = ((H{i-1} - K{i}) .* del_H{i}) .* dZ{i};
        U1_grad = U1_grad + del_Z{i} * transpose(input{i});
        U2_grad = U2_grad + del_Z{i} * transpose(H{i-1});
        Bz_grad = Bz_grad + sum(del_Z{i}, 2);
                       
        del_R{i} = ((W2*H{i-1}) .* del_K{i}) .* dR{i};
        V1_grad = V1_grad + del_R{i} * transpose(input{i});
        V2_grad = V2_grad + del_R{i} * transpose(H{i-1});
        Br_grad = Br_grad + sum(del_R{i}, 2);
        
    end
    
    % at time t = 1
    del_H{1} = (U2' * del_Z{2} + V2' * del_R{2} + Z{2} .* del_H{2} + W2' * (R{2} .* del_K{2})) .* dH{1};
    
    del_K{1} = ((1-Z{1}) .* del_H{1}) .* dK{1};
    W1_grad = W1_grad + del_K{1} * transpose(input{1});
    W2_grad = W2_grad + (R{1} .* del_K{1}) * transpose(H0);
    Bk_grad = Bk_grad + sum(del_K{1}, 2);
           
    del_Z{1} =  ((H0 -1 * K{1}) .* del_H{1}) .* dZ{1};
    U1_grad = U1_grad + del_Z{1} * transpose(input{1});
    U2_grad = U2_grad + del_Z{1} * transpose(H0);
    Bz_grad = Bz_grad + sum(del_Z{1}, 2);
    
        
    del_R{1} = ((W2*H0) .* del_K{1}) .* dR{1};
    V1_grad = V1_grad + del_R{1} * transpose(input{1});
    V2_grad = V2_grad + del_R{1} * transpose(H0);
    Br_grad = Br_grad + sum(del_R{1}, 2);
    
    
    W1_grad = W1_grad / batch_size;
    W2_grad = W2_grad / batch_size;
    W3_grad = W3_grad / batch_size;
    U1_grad = U1_grad / batch_size;
    U2_grad = U2_grad / batch_size;
    V1_grad = V1_grad / batch_size;
    V2_grad = V2_grad / batch_size;
    Bk_grad = Bk_grad / batch_size;
    Bz_grad = Bz_grad / batch_size;
    Br_grad = Br_grad / batch_size;
    Bo_grad = Bo_grad / batch_size;
    
    % gradient clipping
    W1_grad = w_clip(W1_grad, W1, lamda, gc);
    W2_grad = w_clip(W2_grad, W2, lamda, gc);
    W3_grad = w_clip(W3_grad, W3, lamda, gc);
    U1_grad = w_clip(U1_grad, U1, lamda, gc);
    U2_grad = w_clip(U2_grad, U2, lamda, gc);
    V1_grad = w_clip(V1_grad, V1, lamda, gc);
    V2_grad = w_clip(V2_grad, V2, lamda, gc);
    Bk_grad = b_clip(Bk_grad, gc);
    Bz_grad = b_clip(Bz_grad, gc);
    Br_grad = b_clip(Br_grad, gc);
    Bo_grad = b_clip(Bo_grad, gc);
    
    grad = [W1_grad(:); W2_grad(:); U1_grad(:); U2_grad(:); V1_grad(:); V2_grad(:); W3_grad(:); Bk_grad(:); Bz_grad(:); Br_grad(:); Bo_grad(:)];
        
end



