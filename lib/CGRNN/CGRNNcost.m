function [cost, grad] = CGRNNcost(theta, input, output, hidden_size, lamda, gc)
input_size = size(input{1}, 1);
no_repeat = length(input);
output_size = size(output, 1);
batch_size = size(output, 2);

% params
limit0 = 1;
limit1 = hidden_size * input_size;                      % Wi
limit2 = limit1 + hidden_size * hidden_size;            % Wh
limit3 = limit2 + hidden_size * input_size;             % V1
limit4 = limit3 + hidden_size * hidden_size;            % V2
limit5 = limit4 + output_size * hidden_size;            % Wo
limit6 = limit5 + hidden_size;                          % Bk
limit7 = limit6 + hidden_size;                          % Bz
limit8 = limit7 + output_size;                          % Bo

Wi = reshape(theta(limit0:limit1), [hidden_size, input_size]);
Wh = reshape(theta((limit1+1):limit2), [hidden_size, hidden_size]);
V1 = reshape(theta((limit2+1):limit3), [hidden_size, input_size]);
V2 = reshape(theta((limit3+1):limit4), [hidden_size, hidden_size]);
Wo = reshape(theta((limit4+1):limit5), [output_size, hidden_size]);
Bk = reshape(theta((limit5+1):limit6), [hidden_size, 1]);
Bz = reshape(theta((limit6+1):limit7), [hidden_size, 1]);
Bo = reshape(theta((limit7+1):limit8), [output_size, 1]);

% forward pass
H0 = repmat(zeros(hidden_size, 1), 1, batch_size);
% H0 = normrnd(0, 0.2, hidden_size, batch_size);
for i=1:no_repeat
    if i==1
        Hp = H0;
    else
        Hp = H{i-1};
    end
        
    Z_z{i} = V1 * input{i} + V2 * Hp + Bz * ones(1, batch_size);
    [Z{i}, dZ{i}] = ptanh_act(Z_z{i});
    K_z{i} = Wi * input{i} + Wh * Hp + Bk * ones(1, batch_size);
    [K{i}, dK{i}] = tanh_act(K_z{i});
    H_z{i} = Z{i} .* Hp + (1-Z{i}) .* K{i};
    [H{i}, dH{i}] = linear_act(H_z{i});
        
    clear R_z{i} Z_z{i} K_z{i} H_z{i} O_z{i};    
end
O_z = Wo * H{no_repeat} + Bo * ones(1, batch_size);
O_z = O_z - repmat(max(O_z, [], 1), output_size,1);
O = exp(O_z) ./ repmat(sum(exp(O_z)), output_size,1);

loss = -1 * sum(sum(output .* log(O))) / batch_size;

weight_decay = 0.5 * lamda * (sum(sum(Wi .^ 2)) + sum(sum(Wh .^ 2)) + sum(sum(Wo .^ 2)) + sum(sum(V1 .^ 2)) + sum(sum(V2 .^ 2)));

cost = loss + weight_decay;

% backward pass
if nargout > 1
    Wi_grad = zeros(hidden_size, input_size);
    Wh_grad = zeros(hidden_size, hidden_size);
    V1_grad = zeros(hidden_size, input_size);
    V2_grad = zeros(hidden_size, hidden_size);    
    Bk_grad = zeros(hidden_size, 1);
    Bz_grad = zeros(hidden_size, 1);
    Bh_grad = zeros(hidden_size, 1);    
    
    del_O = -1 * (output - O); 
    Wo_grad = del_O * transpose(H{no_repeat}); 
    Bo_grad = sum(del_O, 2);
    
    for i = no_repeat:-1:1
        if i==1
            Hp = H0;
        else
            Hp = H{i-1};
        end        
        
        if i == no_repeat
            del_H{i} = (Wo' * del_O) .* dH{i};
        else
            del_H{i} = (V2' * del_Z{i+1} + Z{i+1} .* del_H{i+1} + Wh' * del_K{i+1}) .* dH{i};
        end
                
        del_K{i} = ((1-Z{i}) .* del_H{i}) .* dK{i};
        Wi_grad = Wi_grad + del_K{i} * transpose(input{i});        
        Wh_grad = Wh_grad + del_K{i} * transpose(Hp);
        Bk_grad = Bk_grad + sum(del_K{i}, 2);
                      
        del_Z{i} = ((Hp - K{i}) .* del_H{i}) .* dZ{i};
        V1_grad = V1_grad + del_Z{i} * transpose(input{i});
        V2_grad = V2_grad + del_Z{i} * transpose(Hp);
        Bz_grad = Bz_grad + sum(del_Z{i}, 2);                       
        
    end
        
    Wi_grad = Wi_grad / batch_size;    
    Wh_grad = Wh_grad / batch_size; 
    Wo_grad = Wo_grad / batch_size;
    V1_grad = V1_grad / batch_size;
    V2_grad = V2_grad / batch_size;
    Bk_grad = Bk_grad / batch_size;
    Bz_grad = Bz_grad / batch_size;
    Bh_grad = Bh_grad / batch_size;
    Bo_grad = Bo_grad / batch_size;
        
    % gradient clipping
    Wi_grad = w_clip(Wi_grad, Wi, lamda, gc);    
    Wh_grad = w_clip(Wh_grad, Wh, lamda, gc);  
    Wo_grad = w_clip(Wo_grad, Wo, lamda, gc);
    V1_grad = w_clip(V1_grad, V1, lamda, gc);
    V2_grad = w_clip(V2_grad, V2, lamda, gc);    
    Bk_grad = b_clip(Bk_grad, gc);
    Bz_grad = b_clip(Bz_grad, gc);
    Bh_grad = b_clip(Bh_grad, gc);
    Bo_grad = b_clip(Bo_grad, gc);    
    
    grad = [Wi_grad(:); Wh_grad(:); V1_grad(:); V2_grad(:); Wo_grad(:); Bk_grad(:); Bz_grad(:); Bh_grad(:); Bo_grad(:)];
        
end



