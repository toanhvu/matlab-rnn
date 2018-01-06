function [cost, grad] = RNN2Lcost(theta, input, output, ha_size, hb_size, lamda, gc)
% theta
theta1 = theta.theta1;
theta2 = theta.theta2;
theta0 = theta.theta0;

% network info, more
input_size = size(input{1}, 1);
no_loop = length(input);
output_size = size(output, 1);
batch_size = size(output, 2);

% params of layer 1
limit0 = 1;
limit1 = ha_size * input_size;
limit2 = limit1 + ha_size * ha_size;
limit3 = limit2 + ha_size;

Wa1 = reshape(theta1(limit0:limit1), [ha_size, input_size]);
Wa2 = reshape(theta1((limit1 +1):limit2), [ha_size, ha_size]);
Ba = reshape(theta1((limit2 +1):limit3), [ha_size, 1]);

% params of layer 2
limit0 = 1;
limit1 = hb_size * ha_size;
limit2 = limit1 + hb_size * hb_size;
limit3 = limit2 + hb_size;

Wb1 = reshape(theta2(limit0:limit1), [hb_size, ha_size]);
Wb2 = reshape(theta2((limit1 +1):limit2), [hb_size, hb_size]);
Bb = reshape(theta2((limit2 +1):limit3), [hb_size, 1]);

% params of output
limit0 = 1;
limit1 = output_size * hb_size;
limit2 = limit1 + output_size;
Wo = reshape(theta0(limit0:limit1), [output_size, hb_size]);
Bo = reshape(theta0((limit1+1):limit2), [output_size, 1]);


% FORWARD pass
for i =1:no_loop
    if i ==1
        Hap = repmat(zeros(ha_size, 1), 1, batch_size);
        Hbp = repmat(zeros(hb_size, 1), 1, batch_size);
    else
        Hap = Ha{i-1};
        Hbp = Hb{i-1};
    end
    Ha_z{i} = Wa1 * input{i} + Wa2 * Hap + Ba * ones(1, batch_size);
    [Ha{i}, dHa{i}] = tanh_act(Ha_z{i});
    
    Hb_z{i} = Wb1 * Ha{i} + Wb2 * Hbp + Bb * ones(1, batch_size);
    [Hb{i}, dHb{i}] = tanh_act(Hb_z{i});

    clear Ha_z{i} Hb_z{i};
end

O_z = Wo * Hb{no_loop} + Bo * ones(1, batch_size);
O_z = O_z - repmat(max(O_z, [], 1), output_size,1);
O = exp(O_z) ./ repmat(sum(exp(O_z)), output_size,1);

loss = -1 * sum(sum(output .* log(O))) / batch_size;
weight_decay = 0.5 * lamda * (sum(sum(Wa1 .^ 2)) + sum(sum(Wa2 .^ 2)) + sum(sum(Wb1 .^ 2)) + sum(sum(Wb2 .^ 2)) + sum(sum(Wo .^ 2)));

cost = loss + weight_decay;


% BACKWARD pass
if nargout >1
    Wa1_grad = zeros(ha_size, input_size);
    Wa2_grad = zeros(ha_size, ha_size);
    Ba_grad = zeros(ha_size, 1);
    
    Wb1_grad = zeros(hb_size, ha_size);
    Wb2_grad = zeros(hb_size, hb_size);    
    Bb_grad = zeros(hb_size, 1);
    
    del_O = -1 * (output - O); 
    Wo_grad = del_O * transpose(Hb{no_loop}); 
    Bo_grad = sum(del_O, 2);
    
    for i=no_loop:-1:1
        if i==1
            Hap = repmat(zeros(ha_size, 1), 1, batch_size);
            Hbp = repmat(zeros(hb_size, 1), 1, batch_size);
        else
            Hap = Ha{i-1};
            Hbp = Hb{i-1};
        end
        
        if i==no_loop
           del_Hb{i}  = (Wo' * del_O) .* dHb{i};
        else
            del_Hb{i} = (Wb2' * del_Hb{i+1}) .* dHb{i};
        end
        
        Wb1_grad = Wb1_grad + del_Hb{i} * transpose(Ha{i});
        Wb2_grad = Wb2_grad + del_Hb{i} * transpose(Hbp);
        Bb_grad = Bb_grad + sum(del_Hb{i}, 2);
        
        
        if i==no_loop
            del_Ha{i} = (Wb1' * del_Hb{i}) .* dHa{i};
        else
            del_Ha{i} = (Wb1' * del_Hb{i} + Wa2' * del_Ha{i+1}) .* dHa{i};
        end
        
        Wa1_grad = Wa1_grad + del_Ha{i} * transpose(input{i});
        Wa2_grad = Wa2_grad + del_Ha{i} * transpose(Hap);
        Ba_grad = Ba_grad + sum(del_Ha{i}, 2);
        
    end
    
    Wa1_grad = Wa1_grad / batch_size;
    Wa2_grad = Wa2_grad / batch_size;
    Ba_grad = Ba_grad / batch_size;
        
    Wb1_grad = Wb1_grad / batch_size;
    Wb2_grad = Wb2_grad / batch_size;
    Bb_grad = Bb_grad / batch_size;
        
    Wo_grad = Wo_grad / batch_size;
    Bo_grad = Bo_grad / batch_size;
    
    % gradient clipping
    Wa1_grad = w_clip(Wa1_grad, Wa1, lamda, gc);
    Wa2_grad = w_clip(Wa2_grad, Wa2, lamda, gc);    
    Ba_grad = b_clip(Ba_grad, gc);
        
    Wb1_grad = w_clip(Wb1_grad, Wb1, lamda, gc);
    Wb2_grad = w_clip(Wb2_grad, Wb2, lamda, gc);
    Bb_grad = b_clip(Bb_grad, gc);   
        
    Wo_grad = w_clip(Wo_grad, Wo, lamda, gc);
    Bo_grad = b_clip(Bo_grad, gc);
    
    theta1_grad = [Wa1_grad(:); Wa2_grad(:); Ba_grad(:)];
    theta2_grad = [Wb1_grad(:); Wb2_grad(:); Bb_grad(:)];
    theta0_grad = [Wo_grad(:); Bo_grad(:)];
    
    grad = struct;
    grad.theta0_grad = theta0_grad;
    grad.theta1_grad = theta1_grad;
    grad.theta2_grad = theta2_grad;
    
end










