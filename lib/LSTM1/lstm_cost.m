function [cost, grad] = lstm_cost(theta, input, output, hidden_size, lamda, gc)
% theta
theta_x = theta.theta_x;
theta_i = theta.theta_i;
theta_f = theta.theta_f;
theta_o = theta.theta_o;
theta0 = theta.theta0;

% network info
input_size =  size(input{1}, 1);
output_size = size(output, 1);
batch_size = size(output, 2);
no_loop = length(input);

%% params
%% lmstm params
% input
limit0 = 1;
limit1 = hidden_size * input_size;
limit2 = limit1 + hidden_size * hidden_size;
limit3 = limit2 + hidden_size;

Wxc = reshape(theta_x(limit0:limit1), [hidden_size, input_size]);
Whc = reshape(theta_x((limit1 +1):limit2), [hidden_size, hidden_size]);
Bc = reshape(theta_x((limit2 +1):limit3), [hidden_size, 1]);

% input gate
limit0 = 1;
limit1 = hidden_size * input_size;
limit2 = limit1 + hidden_size * hidden_size;
limit3 = limit2 + hidden_size * hidden_size;
limit4 = limit3 + hidden_size;

Wxi = reshape(theta_i(limit0:limit1), [hidden_size, input_size]);
Whi = reshape(theta_i((limit1 +1):limit2), [hidden_size, hidden_size]);
Wci = reshape(theta_i((limit2 +1):limit3), [hidden_size, hidden_size]);
Bi = reshape(theta_i((limit3 +1):limit4), [hidden_size, 1]);

% forget gate
Wxf = reshape(theta_f(limit0:limit1), [hidden_size, input_size]);
Whf = reshape(theta_f((limit1 +1):limit2), [hidden_size, hidden_size]);
Wcf = reshape(theta_f((limit2 +1):limit3), [hidden_size, hidden_size]);
Bf = reshape(theta_f((limit3 +1):limit4), [hidden_size, 1]);

% output_gate
Wxo = reshape(theta_o(limit0:limit1), [hidden_size, input_size]);
Who = reshape(theta_o((limit1 +1):limit2), [hidden_size, hidden_size]);
Wco = reshape(theta_o((limit2 +1):limit3), [hidden_size, hidden_size]);
Bo = reshape(theta_o((limit3 +1):limit4), [hidden_size, 1]);

%% output params
limit0 = 1;
limit1 = output_size * hidden_size;
limit2 = limit1 + output_size;

W0 = reshape(theta0(limit0:limit1), [output_size, hidden_size]);
B0 = reshape(theta0((limit1 +1):limit2), [output_size, 1]);

%% forward pass
for i =1:no_loop
    if i==1
        Cp = repmat(zeros(hidden_size, 1), 1, batch_size);
        Hp = repmat(zeros(hidden_size, 1), 1, batch_size);
    else
        Cp = C{i-1};
        Hp = H{i-1};
    end
    
    I_z{i} = Wxi * input{i} + Wci * Cp + Whi * Hp + Bi * ones(1, batch_size);
    [I{i}, dI{i}] = sigmoid(I_z{i});
    clear I_z{i};
    
    F_z{i} = Wxf * input{i} + Wcf * Cp + Whf * Hp + Bf * ones(1, batch_size);
    [F{i}, dF{i}] = sigmoid(F_z{i});
    clear F_z{i};
    
    K_z{i} = Wxc * input{i} + Whc * Hp + Bc * ones(1, batch_size);
    [K{i}, dK{i}] = tanh_act(K_z{i});
    clear K_z{i};
    
    C{i} = F{i} .* Cp + I{i} .* K{i};
    [CC{i}, dCC{i}] = tanh_act(C{i});    
    
    O_z{i} = Wxo * input{i} + Who * Hp + Wco * C{i} + Bo * ones(1, batch_size);
    [O{i}, dO{i}] = sigmoid(O_z{i});
    clear O_z{i};
    
    H{i} = O{i} .* CC{i};   
end

L_z = W0 * H{no_loop} + B0 * ones(1, batch_size);
L_z = L_z - repmat(max(L_z, [], 1), output_size,1);
L = exp(L_z) ./ repmat(sum(exp(L_z)), output_size,1);

loss = -1 * sum(sum(output .* log(L))) / batch_size;

weight_decay = 0.5 * lamda * ( sum(sum(Wxc .^ 2)) + sum(sum(Whc .^ 2)) ...
                + sum(sum(Wxi.^2)) + sum(sum(Whi.^2)) + sum(sum(Wci.^2))...
                + sum(sum(Wxf.^2)) + sum(sum(Whf.^2)) + sum(sum(Wcf.^2))...
                + sum(sum(Wxo.^2)) + sum(sum(Who.^2)) + sum(sum(Wco.^2))...
                + sum(sum(W0.^2)));

cost = loss + weight_decay;

if nargout >1
    Wxc_grad = zeros(hidden_size, input_size);
    Whc_grad = zeros(hidden_size, hidden_size);
    Bc_grad = zeros(hidden_size, 1);
    
    Wxi_grad = zeros(hidden_size, input_size);
    Whi_grad = zeros(hidden_size, hidden_size);
    Wci_grad = zeros(hidden_size, hidden_size);
    Bi_grad = zeros(hidden_size, 1);
    
    Wxf_grad = zeros(hidden_size, input_size);
    Whf_grad = zeros(hidden_size, hidden_size);
    Wcf_grad = zeros(hidden_size, hidden_size);
    Bf_grad = zeros(hidden_size, 1);
    
    Wxo_grad = zeros(hidden_size, input_size);
    Who_grad = zeros(hidden_size, hidden_size);
    Wco_grad = zeros(hidden_size, hidden_size);
    Bo_grad = zeros(hidden_size, 1);
    
    del_L = -1 * (output - L); 
    W0_grad = del_L * transpose(H{no_loop}); 
    B0_grad = sum(del_L, 2);
    
    for i=no_loop:-1:1
        if i==1
            Cp = repmat(zeros(hidden_size, 1), 1, batch_size);
            Hp = repmat(zeros(hidden_size, 1), 1, batch_size);
        else
            Cp = C{i-1};
            Hp = H{i-1};            
        end
        
        if i==no_loop
            del_H{i} = W0' * del_L;
            del_O{i} = (CC{i} .* del_H{i}) .* dO{i};        
            del_C{i} = (O{i} .* del_H{i}) .* dCC{i};
        else
            del_H{i} = Whi' * del_I{i+1} + Whf' * del_F{i+1} + Whc' * del_K{i+1} + Who' * del_O{i+1};
            del_O{i} = (CC{i} .* del_H{i}) .* dO{i};        
            del_C{i} = (O{i} .* del_H{i}) .* dCC{i} + Wco' * del_O{i} + F{i+1}.* del_C{i+1} + Wcf' * del_F{i+1} + Wci' * del_I{i+1};
        end
        del_K{i} = (I{i} .* del_C{i}) .* dK{i};
        del_F{i} = (Cp .* del_C{i}) .* dF{i};
        del_I{i} = (K{i} .* del_C{i}) .* dI{i};
        
        Wxo_grad = Wxo_grad + del_O{i} * transpose(input{i});
        Who_grad = Who_grad + del_O{i} * transpose(Hp);
        Wco_grad = Wco_grad + del_O{i} * transpose(C{i});
        Bo_grad = Bo_grad + sum(del_O{i}, 2);
        
        Wxf_grad = Wxf_grad + del_F{i} * transpose(input{i});
        Whf_grad = Whf_grad + del_F{i} * transpose(Hp);
        Wcf_grad = Wcf_grad + del_F{i} * transpose(Cp);
        Bf_grad = Bf_grad + sum(del_F{i}, 2);
        
        Wxi_grad = Wxi_grad + del_I{i} * transpose(input{i});
        Whi_grad = Whi_grad + del_I{i} * transpose(Hp);
        Wci_grad = Wci_grad + del_I{i} * transpose(Cp);
        Bi_grad = Bi_grad + sum(del_I{i}, 2);
        
        Wxc_grad = Wxc_grad + del_K{i} * transpose(input{i});
        Whc_grad = Whc_grad + del_K{i} * transpose(Hp);
        Bc_grad = Bc_grad + sum(del_K{i}, 2);       
        
    end
    
    % / batch_size
    Wxc_grad = Wxc_grad / batch_size;
    Whc_grad = Whc_grad / batch_size;
    Bc_grad = Bc_grad / batch_size;
    
    Wxi_grad = Wxi_grad / batch_size;
    Whi_grad = Whi_grad / batch_size;
    Wci_grad = Wci_grad / batch_size;
    Bi_grad = Bi_grad / batch_size;
    
    Wxf_grad = Wxf_grad / batch_size;
    Whf_grad = Whf_grad / batch_size;
    Wcf_grad = Wcf_grad / batch_size;
    Bf_grad = Bf_grad / batch_size;
    
    Wxo_grad = Wxo_grad / batch_size;
    Who_grad = Who_grad / batch_size;
    Wco_grad = Wco_grad / batch_size;
    Bo_grad = Bo_grad / batch_size;
    
    W0_grad = W0_grad / batch_size;
    B0_grad = B0_grad / batch_size;
    
    % gradient clipping
    Wxc_grad = w_clip(Wxc_grad, Wxc, lamda, gc);
    Whc_grad = w_clip(Whc_grad, Whc, lamda, gc);
    Bc_grad = b_clip(Bc_grad, gc);
    
    Wxi_grad = w_clip(Wxi_grad, Wxi, lamda, gc);
    Whi_grad = w_clip(Whi_grad, Whi, lamda, gc);
    Wci_grad = w_clip(Wci_grad, Wci, lamda, gc);
    Bi_grad = b_clip(Bi_grad, gc);
    
    Wxf_grad = w_clip(Wxf_grad, Wxf, lamda, gc);
    Whf_grad = w_clip(Whf_grad, Whf, lamda, gc);
    Wcf_grad = w_clip(Wcf_grad, Wcf, lamda, gc);
    Bf_grad = b_clip(Bf_grad, gc);
    
    Wxo_grad = w_clip(Wxo_grad, Wxo, lamda, gc);
    Who_grad = w_clip(Who_grad, Who, lamda, gc);
    Wco_grad = w_clip(Wco_grad, Wco, lamda, gc);
    Bo_grad = b_clip(Bo_grad, gc);
    
    W0_grad = w_clip(W0_grad, W0, lamda, gc);
    B0_grad = b_clip(B0_grad, gc);   
    
    grad_theta0 = [W0_grad(:); B0_grad(:)];
    grad_theta_i = [Wxi_grad(:); Whi_grad(:); Wci_grad(:); Bi_grad(:)];
    grad_theta_f = [Wxf_grad(:); Whf_grad(:); Wcf_grad(:) ;Bf_grad(:)];
    grad_theta_o = [Wxo_grad(:); Who_grad(:); Wco_grad(:) ;Bo_grad(:)];
    grad_theta_x = [Wxc_grad(:); Whc_grad(:); Bc_grad(:)];
    
    grad = struct;
    grad.grad_theta0 = grad_theta0;
    grad.grad_theta_i = grad_theta_i;
    grad.grad_theta_f = grad_theta_f;
    grad.grad_theta_o = grad_theta_o;
    grad.grad_theta_x = grad_theta_x;
end         

