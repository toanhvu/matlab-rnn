function [cost, grad] = DIFNETcost(theta, input, output, hidden_size, lamda, beta, gc)
input_size = size(input{1}, 1);
no_repeat = length(input);
output_size = size(output, 1);
batch_size = size(output, 2);

% params
limit0 = 1;
limit1 = input_size * hidden_size;
limit2 = limit1 + hidden_size * input_size;
limit3 = limit2 + hidden_size * hidden_size;
limit4 = limit3 + hidden_size * input_size;
limit5 = limit4 + output_size * hidden_size;
limit6 = limit5 + input_size;
limit7 = limit6 + hidden_size;
limit8 = limit7 + hidden_size;
limit9 = limit8 + output_size;

Wp = reshape(theta(limit0:limit1), [input_size, hidden_size]);
Wk = reshape(theta((limit1+1):limit2), [hidden_size, input_size]);
Wh = reshape(theta((limit2+1):limit3), [hidden_size, hidden_size]);
Wz = reshape(theta((limit3+1):limit4), [hidden_size, input_size]);
Wo = reshape(theta((limit4+1):limit5), [output_size, hidden_size]);
Bp = reshape(theta((limit5+1):limit6), [input_size, 1]);
Bk = reshape(theta((limit6+1):limit7), [hidden_size, 1]);
Bz = reshape(theta((limit7+1):limit8), [hidden_size, 1]);
Bo = reshape(theta((limit8+1):limit9), [output_size, 1]);

%% forward
MSE = 0;
thre_indx = 1;

for i=1:no_repeat
    if i==1
       Pp = repmat(zeros(input_size, 1), 1, batch_size);
       Hp = repmat(zeros(hidden_size, 1), 1, batch_size);
    else
       Pp = P{i-1};
       Hp = H{i-1};
    end
    diff{i} = input{i} - Pp;
    if i>thre_indx
        MSE = MSE + 1/2 * sum(sum(diff{i}.^2)) / batch_size;
    end
    K_z{i} = Wk * diff{i} + Wh * Hp + Bk * ones(1, batch_size);
    [K{i}, dK{i}] = relu(K_z{i});
    
    Z_z{i} = Wz * diff{i} + Bz * ones(1, batch_size);
    [Z{i}, dZ{i}] = sigmoid(Z_z{i});
    
    H_z{i} = Z{i} .* Hp + (1 - Z{i}) .* K{i};
    [H{i}, dH{i}] = linear_act(H_z{i});
    
    P_z{i} = Wp * H{i} + Bp * ones(1, batch_size);
    [P{i}, dP{i}] = linear_act(P_z{i});
    
    clear K_z{i} Z_z{i} H_z{i} P_z{i};    
    
end
O_z = Wo * H{no_repeat} + Bo * ones(1, batch_size);
O_z = O_z - repmat(max(O_z, [], 1), output_size,1);
O = exp(O_z) ./ repmat(sum(exp(O_z)), output_size,1);

loss = -1 * sum(sum(output .* log(O))) / batch_size;
weight_decay = 0.5 * lamda * (sum(sum(Wp .^ 2)) + sum(sum(Wk .^ 2)) + sum(sum(Wh .^ 2)) + sum(sum(Wz .^ 2)) + sum(sum(Wo .^ 2)));

cost = loss + weight_decay + beta/no_repeat * MSE;

%% backward
if nargout >1
    Wp_grad = zeros(input_size, hidden_size);
    Wk_grad = zeros(hidden_size, input_size);
    Wh_grad = zeros(hidden_size, hidden_size);
    Wz_grad = zeros(hidden_size, input_size);
    Bp_grad = zeros(input_size, 1);
    Bk_grad = zeros(hidden_size, 1);
    Bz_grad = zeros(hidden_size, 1);
        
    % at time t = no_repeat
    del_O = -1 * (output - O); 
    Wo_grad = del_O * transpose(H{no_repeat}); 
    Bo_grad = sum(del_O, 2);
    
    for i=no_repeat:-1:1
        if i==1            
            Hp = repmat(zeros(hidden_size, 1), 1, batch_size);
        else            
            Hp = H{i-1};
        end
        
        if i==no_repeat
            del_H{i} = (Wo' * del_O) .* dH{i};                          
        else            
            del_H{i} = (Wp' * del_P{i} + Wh' * del_K{i+1} + Z{i+1} .* del_H{i+1}) .* dH{i};                   
        end
                
        del_K{i} = (1-Z{i}) .* del_H{i} .* dK{i};        
        Wk_grad = Wk_grad + del_K{i} * transpose(diff{i});
        Wh_grad = Wh_grad + del_K{i} * transpose(Hp);
        Bk_grad = Bk_grad + sum(del_K{i}, 2);
        
        del_Z{i} = (Hp - K{i}) .* del_H{i} .* dZ{i};
        Wz_grad = Wz_grad + del_Z{i} * transpose(diff{i});
        Bz_grad = Bz_grad + sum(del_Z{i}, 2);
        
        if i > thre_indx
            del_P{i-1} = (-Wk' * del_K{i} - Wz' * del_Z{i} - beta/no_repeat * diff{i}) .* dP{i-1};
            Wp_grad = Wp_grad + del_P{i-1} * transpose(Hp);
            Bp_grad = Bp_grad + sum(del_P{i-1}, 2);
        elseif i<=thre_indx && i>1            
            del_P{i-1} = (-Wk' * del_K{i} - Wz' * del_Z{i}) .* dP{i-1};
            Wp_grad = Wp_grad + del_P{i-1} * transpose(Hp);
            Bp_grad = Bp_grad + sum(del_P{i-1}, 2);
        end        
    end
    
    % / batch_size
    Wp_grad = Wp_grad / batch_size;
    Wk_grad = Wk_grad / batch_size;
    Wh_grad = Wh_grad / batch_size;
    Wz_grad = Wz_grad / batch_size;
    Wo_grad = Wo_grad / batch_size;
    Bp_grad = Bp_grad / batch_size;
    Bk_grad = Bk_grad / batch_size;
    Bz_grad = Bz_grad / batch_size;
    Bo_grad = Bo_grad / batch_size;
    
    % gradient clipping
    Wp_grad = w_clip(Wp_grad, Wp, lamda, gc);
    Wk_grad = w_clip(Wk_grad, Wk, lamda, gc);
    Wh_grad = w_clip(Wh_grad, Wh, lamda, gc);
    Wz_grad = w_clip(Wz_grad, Wz, lamda, gc);
    Wo_grad = w_clip(Wo_grad, Wo, lamda, gc);
    Bp_grad = b_clip(Bp_grad, gc);
    Bk_grad = b_clip(Bk_grad, gc);
    Bz_grad = b_clip(Bz_grad, gc);    
    Bo_grad = b_clip(Bo_grad, gc);
    
    grad = [Wp_grad(:); Wk_grad(:); Wh_grad(:); Wz_grad(:); Wo_grad(:); Bp_grad(:); Bk_grad(:); Bz_grad(:); Bo_grad(:)];
    
end




















