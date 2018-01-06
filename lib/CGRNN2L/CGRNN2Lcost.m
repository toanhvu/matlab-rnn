function [cost, grad] = CGRNN2Lcost(theta, input, output, ha_size, hb_size, lamda, gc)
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
limit3 = limit2 + ha_size * input_size;
limit4 = limit3 + ha_size * ha_size;
limit5 = limit4 + ha_size;
limit6 = limit5 + ha_size;

Wa1 = reshape(theta1(limit0:limit1), [ha_size, input_size]);
Wa2 = reshape(theta1((limit1 +1):limit2), [ha_size, ha_size]);
Va1 = reshape(theta1((limit2 +1):limit3), [ha_size, input_size]);
Va2 = reshape(theta1((limit3 +1):limit4), [ha_size, ha_size]);
Bka = reshape(theta1((limit4 +1):limit5), [ha_size, 1]);
Bza = reshape(theta1((limit5 +1):limit6), [ha_size, 1]);

% params of layer 2
limit0 = 1;
limit1 = hb_size * ha_size;
limit2 = limit1 + hb_size * hb_size;
limit3 = limit2 + hb_size * ha_size;
limit4 = limit3 + hb_size * hb_size;
limit5 = limit4 + hb_size;
limit6 = limit5 + hb_size;

Wb1 = reshape(theta2(limit0:limit1), [hb_size, ha_size]);
Wb2 = reshape(theta2((limit1 +1):limit2), [hb_size, hb_size]);
Vb1 = reshape(theta2((limit2 +1):limit3), [hb_size, ha_size]);
Vb2 = reshape(theta2((limit3 +1):limit4), [hb_size, hb_size]);
Bkb = reshape(theta2((limit4 +1):limit5), [hb_size, 1]);
Bzb = reshape(theta2((limit5 +1):limit6), [hb_size, 1]);

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
    
    Za_z{i} = Va1 * input{i} + Va2 * Hap + Bza * ones(1, batch_size);
    [Za{i}, dZa{i}] = sigmoid(Za_z{i});
    Ka_z{i} = Wa1 * input{i} + Wa2 * Hap + Bka * ones(1, batch_size);
    [Ka{i}, dKa{i}] = tanh_act(Ka_z{i});
    Ha_z{i} = Za{i} .* Hap + (1 - Za{i}) .* Ka{i};
    [Ha{i}, dHa{i}] = linear_act(Ha_z{i});
    
    Zb_z{i} = Vb1 * Ha{i} + Vb2 * Hbp + Bzb * ones(1, batch_size);
    [Zb{i}, dZb{i}] = sigmoid(Zb_z{i});
    Kb_z{i} = Wb1 * Ha{i} + Wb2 * Hbp + Bkb * ones(1, batch_size);
    [Kb{i}, dKb{i}] = tanh_act(Kb_z{i});
    Hb_z{i} = Zb{i} .* Hbp + (1 - Zb{i}) .* Kb{i};
    [Hb{i}, dHb{i}] = linear_act(Hb_z{i});

    clear Za_z{i} Ka_z{i} Ha_z{i} Zb_z{i} Kb_z{i} Hb_z{i};

end

O_z = Wo * Hb{no_loop} + Bo * ones(1, batch_size);
O_z = O_z - repmat(max(O_z, [], 1), output_size,1);
O = exp(O_z) ./ repmat(sum(exp(O_z)), output_size,1);

loss = -1 * sum(sum(output .* log(O))) / batch_size;
weight_decay = 0.5 * lamda * (sum(sum(Wa1 .^ 2)) + sum(sum(Wa2 .^ 2)) + sum(sum(Va1 .^ 2)) + sum(sum(Va2 .^ 2))...
                                + sum(sum(Wb1 .^ 2)) + sum(sum(Wb2 .^ 2)) + sum(sum(Vb1 .^ 2)) + sum(sum(Vb2 .^ 2))...
                                + sum(sum(Wo .^ 2)));

cost = loss + weight_decay;


% BACKWARD pass
if nargout >1
    Wa1_grad = zeros(ha_size, input_size);
    Wa2_grad = zeros(ha_size, ha_size);
    Va1_grad = zeros(ha_size, input_size);
    Va2_grad = zeros(ha_size, ha_size);
    Bka_grad = zeros(ha_size, 1);
    Bza_grad = zeros(ha_size, 1);
    
    Wb1_grad = zeros(hb_size, ha_size);
    Wb2_grad = zeros(hb_size, hb_size);
    Vb1_grad = zeros(hb_size, ha_size);
    Vb2_grad = zeros(hb_size, hb_size);    
    Bkb_grad = zeros(hb_size, 1);
    Bzb_grad = zeros(hb_size, 1);
    
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
            del_Hb{i} = (Vb2' * del_Zb{i+1} + Zb{i+1} .* del_Hb{i+1} + Wb2' * del_Kb{i+1}) .* dHb{i};
        end
        
        del_Kb{i} = ((1-Zb{i}) .* del_Hb{i}) .* dKb{i};
        Wb1_grad = Wb1_grad + del_Kb{i} * transpose(Ha{i});
        Wb2_grad = Wb2_grad + del_Kb{i} * transpose(Hbp);
        Bkb_grad = Bkb_grad + sum(del_Kb{i}, 2);
        
        del_Zb{i} = ((Hbp - Kb{i}) .* del_Hb{i}) .* dZb{i};
        Vb1_grad = Vb1_grad + del_Zb{i} * transpose(Ha{i});
        Vb2_grad = Vb2_grad + del_Zb{i} * transpose(Hbp);
        Bzb_grad = Bzb_grad + sum(del_Zb{i}, 2);        
        
        if i==no_loop
            del_Ha{i} = (Wb1' * del_Kb{i} + Vb1' * del_Zb{i}) .* dHa{i};
        else
            del_Ha{i} = (Wb1' * del_Kb{i} + Vb1' * del_Zb{i} +...
                        Va2' * del_Za{i+1} + Za{i+1} .* del_Ha{i+1} + Wa2' * del_Ka{i+1}) .* dHa{i};
        end
                
        del_Ka{i} = ((1-Za{i}) .* del_Ha{i}) .* dKa{i};
        Wa1_grad = Wa1_grad + del_Ka{i} * transpose(input{i});
        Wa2_grad = Wa2_grad + del_Ka{i} * transpose(Hap);
        Bka_grad = Bka_grad + sum(del_Ka{i}, 2);
        
        del_Za{i} = ((Hap - Ka{i}) .* del_Ha{i}) .* dZa{i};
        Va1_grad = Va1_grad + del_Za{i} * transpose(input{i});
        Va2_grad = Va2_grad + del_Za{i} * transpose(Hap);
        Bza_grad = Bza_grad + sum(del_Za{i}, 2);
        
    end
    
    Wa1_grad = Wa1_grad / batch_size;
    Wa2_grad = Wa2_grad / batch_size;
    Va1_grad = Va1_grad / batch_size;
    Va2_grad = Va2_grad / batch_size;    
    Bka_grad = Bka_grad / batch_size;
    Bza_grad = Bza_grad / batch_size;
        
    Wb1_grad = Wb1_grad / batch_size;
    Wb2_grad = Wb2_grad / batch_size;
    Vb1_grad = Vb1_grad / batch_size;
    Vb2_grad = Vb2_grad / batch_size;
    Bkb_grad = Bkb_grad / batch_size;
    Bzb_grad = Bzb_grad / batch_size;
        
    Wo_grad = Wo_grad / batch_size;
    Bo_grad = Bo_grad / batch_size;
    
    % gradient clipping
    Wa1_grad = w_clip(Wa1_grad, Wa1, lamda, gc);
    Wa2_grad = w_clip(Wa2_grad, Wa2, lamda, gc);
    Va1_grad = w_clip(Va1_grad, Va1, lamda, gc);
    Va2_grad = w_clip(Va2_grad, Va2, lamda, gc);
    Bka_grad = b_clip(Bka_grad, gc);
    Bza_grad = b_clip(Bza_grad, gc);
        
    Wb1_grad = w_clip(Wb1_grad, Wb1, lamda, gc);
    Wb2_grad = w_clip(Wb2_grad, Wb2, lamda, gc);
    Vb1_grad = w_clip(Vb1_grad, Vb1, lamda, gc);
    Vb2_grad = w_clip(Vb2_grad, Vb2, lamda, gc);    
    Bkb_grad = b_clip(Bkb_grad, gc);
    Bzb_grad = b_clip(Bzb_grad, gc);
        
    Wo_grad = w_clip(Wo_grad, Wo, lamda, gc);
    Bo_grad = b_clip(Bo_grad, gc);
    
    theta1_grad = [Wa1_grad(:); Wa2_grad(:); Va1_grad(:); Va2_grad(:); Bka_grad(:); Bza_grad(:)];
    theta2_grad = [Wb1_grad(:); Wb2_grad(:); Vb1_grad(:); Vb2_grad(:); Bkb_grad(:); Bzb_grad(:)];
    theta0_grad = [Wo_grad(:); Bo_grad(:)];
    
    grad = struct;
    grad.theta0_grad = theta0_grad;
    grad.theta1_grad = theta1_grad;
    grad.theta2_grad = theta2_grad;
    
end










