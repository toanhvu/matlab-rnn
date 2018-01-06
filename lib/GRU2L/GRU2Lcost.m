function [cost, grad] = GRU2Lcost(theta, input, output, ha_size, hb_size, lamda, gc)
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
limit5 = limit4 + ha_size * input_size;
limit6 = limit5 + ha_size * ha_size;
limit7 = limit6 + ha_size;
limit8 = limit7 + ha_size;
limit9 = limit8 + ha_size;

Wa1 = reshape(theta1(limit0:limit1), [ha_size, input_size]);
Wa2 = reshape(theta1((limit1 +1):limit2), [ha_size, ha_size]);
Ua1 = reshape(theta1((limit2 +1):limit3), [ha_size, input_size]);
Ua2 = reshape(theta1((limit3 +1):limit4), [ha_size, ha_size]);
Va1 = reshape(theta1((limit4 +1):limit5), [ha_size, input_size]);
Va2 = reshape(theta1((limit5 +1):limit6), [ha_size, ha_size]);
Bka = reshape(theta1((limit6 +1):limit7), [ha_size, 1]);
Bza = reshape(theta1((limit7 +1):limit8), [ha_size, 1]);
Bra = reshape(theta1((limit8 +1):limit9), [ha_size, 1]);

% params of layer 2
limit0 = 1;
limit1 = hb_size * ha_size;
limit2 = limit1 + hb_size * hb_size;
limit3 = limit2 + hb_size * ha_size;
limit4 = limit3 + hb_size * hb_size;
limit5 = limit4 + hb_size * ha_size;
limit6 = limit5 + hb_size * hb_size;
limit7 = limit6 + hb_size;
limit8 = limit7 + hb_size;
limit9 = limit8 + hb_size;

Wb1 = reshape(theta2(limit0:limit1), [hb_size, ha_size]);
Wb2 = reshape(theta2((limit1 +1):limit2), [hb_size, hb_size]);
Ub1 = reshape(theta2((limit2 +1):limit3), [hb_size, ha_size]);
Ub2 = reshape(theta2((limit3 +1):limit4), [hb_size, hb_size]);
Vb1 = reshape(theta2((limit4 +1):limit5), [hb_size, ha_size]);
Vb2 = reshape(theta2((limit5 +1):limit6), [hb_size, hb_size]);
Bkb = reshape(theta2((limit6 +1):limit7), [hb_size, 1]);
Bzb = reshape(theta2((limit7 +1):limit8), [hb_size, 1]);
Brb = reshape(theta2((limit8 +1):limit9), [hb_size, 1]);

% params of output

limit0 = 1;
limit1 = output_size * hb_size;
limit2 = limit1 + output_size;
Wo = reshape(theta0(limit0:limit1), [output_size, hb_size]);
Bo = reshape(theta0((limit1+1):limit2), [output_size, 1]);


% FORWARD pass
h0_a = zeros(ha_size, 1);
h0_b = zeros(hb_size, 1);

H0_a = repmat(h0_a, 1, batch_size);
H0_b = repmat(h0_b, 1, batch_size);

% at t = 1
for i =1:no_loop
    if i ==1
        Ha0 = H0_a;
        Hb0 = H0_b;
    else
        Ha0 = Ha{i-1};
        Hb0 = Hb{i-1};
    end
    Ra_z{i} = Va1 * input{i} + Va2 * Ha0 + Bra *  ones(1, batch_size);
    [Ra{i}, dRa{i}] = sigmoid(Ra_z{i});
    Za_z{i} = Ua1 * input{i} + Ua2 * Ha0 + Bza * ones(1, batch_size);
    [Za{i}, dZa{i}] = sigmoid(Za_z{i});
    Ka_z{i} = Wa1 * input{i} + Ra{i} .* (Wa2 * Ha0) + Bka * ones(1, batch_size);
    [Ka{i}, dKa{i}] = tanh_act(Ka_z{i});
    Ha_z{i} = Za{i} .* Ha0 + (1 - Za{i}) .* Ka{i};
    [Ha{i}, dHa{i}] = linear_act(Ha_z{i});

    Rb_z{i} = Vb1 * Ha{i} + Vb2 * Hb0 + Brb * ones(1, batch_size);
    [Rb{i}, dRb{i}] = sigmoid(Rb_z{i});
    Zb_z{i} = Ub1 * Ha{i} + Ub2 * Hb0 + Bzb * ones(1, batch_size);
    [Zb{i}, dZb{i}] = sigmoid(Zb_z{i});
    Kb_z{i} = Wb1 * Ha{i} + Rb{i} .* (Wb2 * Hb0) + Bkb * ones(1, batch_size);
    [Kb{i}, dKb{i}] = tanh_act(Kb_z{i});
    Hb_z{i} = Zb{i} .* Hb0 + (1 - Zb{i}) .* Kb{i};
    [Hb{i}, dHb{i}] = linear_act(Hb_z{i});

    clear Ra_z{i} Za_z{i} Ka_z{i} Ha_z{i} Rb_z{i} Zb_z{i} Kb_z{i} Hb_z{i};

end

O_z = Wo * Hb{no_loop} + Bo * ones(1, batch_size);
O_z = O_z - repmat(max(O_z, [], 1), output_size,1);
O = exp(O_z) ./ repmat(sum(exp(O_z)), output_size,1);

loss = -1 * sum(sum(output .* log(O))) / batch_size;
weight_decay = 0.5 * lamda * (sum(sum(Wa1 .^ 2)) + sum(sum(Wa2 .^ 2)) + sum(sum(Ua1 .^ 2)) + sum(sum(Ua2 .^ 2)) + sum(sum(Va1 .^ 2)) + sum(sum(Va2 .^ 2))...
                                + sum(sum(Wb1 .^ 2)) + sum(sum(Wb2 .^ 2)) + sum(sum(Ub1 .^ 2)) + sum(sum(Ub2 .^ 2)) + sum(sum(Vb1 .^ 2)) + sum(sum(Vb2 .^ 2))...
                                + sum(sum(Wo .^ 2)));

cost = loss + weight_decay;


% BACKWARD pass
if nargout >1
    Wa1_grad = zeros(ha_size, input_size);
    Wa2_grad = zeros(ha_size, ha_size);
    Va1_grad = zeros(ha_size, input_size);
    Va2_grad = zeros(ha_size, ha_size);
    Ua1_grad = zeros(ha_size, input_size);
    Ua2_grad = zeros(ha_size, ha_size);
    Bka_grad = zeros(ha_size, 1);
    Bza_grad = zeros(ha_size, 1);
    Bra_grad = zeros(ha_size, 1);

    Wb1_grad = zeros(hb_size, ha_size);
    Wb2_grad = zeros(hb_size, hb_size);
    Vb1_grad = zeros(hb_size, ha_size);
    Vb2_grad = zeros(hb_size, hb_size);
    Ub1_grad = zeros(hb_size, ha_size);
    Ub2_grad = zeros(hb_size, hb_size);
    Bkb_grad = zeros(hb_size, 1);
    Bzb_grad = zeros(hb_size, 1);
    Brb_grad = zeros(hb_size, 1);
    
    del_O = -1 * (output - O); 
    Wo_grad = del_O * transpose(Hb{no_loop}); 
    Bo_grad = sum(del_O, 2);
    
    for i=no_loop:-1:1
        if i==no_loop
           del_Hb{i}  = (Wo' * del_O) .* dHb{i};
        else
            del_Hb{i} = (Ub2' * del_Zb{i+1} + Vb2' * del_Rb{i+1} + Zb{i+1} .* del_Hb{i+1} + Wb2' * (Rb{i+1} .* del_Kb{i+1})) .* dHb{i};
        end
        
        if i==1
            Hap = H0_a;
            Hbp = H0_b;
        else
            Hap = Ha{i-1};
            Hbp = Hb{i-1};
        end
        
        del_Kb{i} = ((1-Zb{i}) .* del_Hb{i}) .* dKb{i};
        Wb1_grad = Wb1_grad + del_Kb{i} * transpose(Ha{i});
        Wb2_grad = Wb2_grad + (Rb{i} .* del_Kb{i}) * transpose(Hbp);
        Bkb_grad = Bkb_grad + sum(del_Kb{i}, 2);
        
        del_Zb{i} = ((Hbp - Kb{i}) .* del_Hb{i}) .* dZb{i};
        Ub1_grad = Ub1_grad + del_Zb{i} * transpose(Ha{i});
        Ub2_grad = Ub2_grad + del_Zb{i} * transpose(Hbp);
        Bzb_grad = Bzb_grad + sum(del_Zb{i}, 2);
        
        del_Rb{i} = ((Wb2*Hbp) .* del_Kb{i}) .* dRb{i};
        Vb1_grad = Vb1_grad + del_Rb{i} * transpose(Ha{i});
        Vb2_grad = Vb2_grad + del_Rb{i} * transpose(Hbp);
        Brb_grad = Brb_grad + sum(del_Rb{i}, 2);
        
        if i==no_loop
            del_Ha{i} = (Wb1' * del_Kb{i} + Ub1' * del_Zb{i} + Vb1' * del_Rb{i}) .* dHa{i};
        else
            del_Ha{i} = (Wb1' * del_Kb{i} + Ub1' * del_Zb{i} + Vb1' * del_Rb{i} +...
                        Ua2' * del_Za{i+1} + Va2' * del_Ra{i+1} + Za{i+1} .* del_Ha{i+1} + Wa2' * (Ra{i+1} .* del_Ka{i+1})) .* dHa{i};
        end
        
        del_Ka{i} = ((1-Za{i}) .* del_Ha{i}) .* dKa{i};
        Wa1_grad = Wa1_grad + del_Ka{i} * transpose(input{i});
        Wa2_grad = Wa2_grad + (Ra{i} .* del_Ka{i}) * transpose(Hap);
        Bka_grad = Bka_grad + sum(del_Ka{i}, 2);
        
        del_Za{i} = ((Hap - Ka{i}) .* del_Ha{i}) .* dZa{i};
        Ua1_grad = Ua1_grad + del_Za{i} * transpose(input{i});
        Ua2_grad = Ua2_grad + del_Za{i} * transpose(Hap);
        Bza_grad = Bza_grad + sum(del_Za{i}, 2);
        
        del_Ra{i} = ((Wa2*Hap) .* del_Ka{i}) .* dRa{i};
        Va1_grad = Va1_grad + del_Ra{i} * transpose(input{i});
        Va2_grad = Va2_grad + del_Ra{i} * transpose(Hap);
        Bra_grad = Bra_grad + sum(del_Ra{i}, 2);
    end
    
    Wa1_grad = Wa1_grad / batch_size;
    Wa2_grad = Wa2_grad / batch_size;
    Ua1_grad = Ua1_grad / batch_size;
    Ua2_grad = Ua2_grad / batch_size;
    Va1_grad = Va1_grad / batch_size;
    Va2_grad = Va2_grad / batch_size;
    Bka_grad = Bka_grad / batch_size;
    Bza_grad = Bza_grad / batch_size;
    Bra_grad = Bra_grad / batch_size;
    
    Wb1_grad = Wb1_grad / batch_size;
    Wb2_grad = Wb2_grad / batch_size;
    Ub1_grad = Ub1_grad / batch_size;
    Ub2_grad = Ub2_grad / batch_size;
    Vb1_grad = Vb1_grad / batch_size;
    Vb2_grad = Vb2_grad / batch_size;
    Bkb_grad = Bkb_grad / batch_size;
    Bzb_grad = Bzb_grad / batch_size;
    Brb_grad = Brb_grad / batch_size;
    
    Wo_grad = Wo_grad / batch_size;
    Bo_grad = Bo_grad / batch_size;
    
    % gradient clipping
    Wa1_grad = w_clip(Wa1_grad, Wa1, lamda, gc);
    Wa2_grad = w_clip(Wa2_grad, Wa2, lamda, gc);
    Ua1_grad = w_clip(Ua1_grad, Ua1, lamda, gc);
    Ua2_grad = w_clip(Ua2_grad, Ua2, lamda, gc);
    Va1_grad = w_clip(Va1_grad, Va1, lamda, gc);
    Va2_grad = w_clip(Va2_grad, Va2, lamda, gc);
    Bka_grad = b_clip(Bka_grad, gc);
    Bza_grad = b_clip(Bza_grad, gc);
    Bra_grad = b_clip(Bra_grad, gc);
    
    Wb1_grad = w_clip(Wb1_grad, Wb1, lamda, gc);
    Wb2_grad = w_clip(Wb2_grad, Wb2, lamda, gc);
    Ub1_grad = w_clip(Ub1_grad, Ub1, lamda, gc);
    Ub2_grad = w_clip(Ub2_grad, Ub2, lamda, gc);
    Vb1_grad = w_clip(Vb1_grad, Vb1, lamda, gc);
    Vb2_grad = w_clip(Vb2_grad, Vb2, lamda, gc);
    Bkb_grad = b_clip(Bkb_grad, gc);
    Bzb_grad = b_clip(Bzb_grad, gc);
    Brb_grad = b_clip(Brb_grad, gc);
    
    Wo_grad = w_clip(Wo_grad, Wo, lamda, gc);
    Bo_grad = b_clip(Bo_grad, gc);
    
    theta1_grad = [Wa1_grad(:); Wa2_grad(:); Ua1_grad(:); Ua2_grad(:); Va1_grad(:); Va2_grad(:); Bka_grad(:); Bza_grad(:); Bra_grad(:)];
    theta2_grad = [Wb1_grad(:); Wb2_grad(:); Ub1_grad(:); Ub2_grad(:); Vb1_grad(:); Vb2_grad(:); Bkb_grad(:); Bzb_grad(:); Brb_grad(:)];
    theta0_grad = [Wo_grad(:); Bo_grad(:)];
    
    grad = struct;
    grad.theta0_grad = theta0_grad;
    grad.theta1_grad = theta1_grad;
    grad.theta2_grad = theta2_grad;
    
end










