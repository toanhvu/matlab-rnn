function theta = CGRNN_init(input_size, hidden_size, output_size)
avg = 0;
sigma = 1; %0.01;

% r = sqrt(6) / sqrt(input_size + hidden_size + output_size); 
% W1 = rand(hidden_size, input_size) * 2 * r - r;
% W2 = rand(hidden_size, hidden_size) * 2 * r - r;
% U1 = rand(hidden_size, input_size) * 2 * r - r;
% U2 = rand(hidden_size, hidden_size) * 2 * r - r;
% V1 = rand(hidden_size, input_size) * 2 * r - r;
% V2 = rand(hidden_size, hidden_size) * 2 * r - r;
% W3 = rand(output_size, hidden_size) * 2 * r - r;

Wi = 0.01 * normrnd(avg,sigma, hidden_size, input_size);
Wh = 0.01 * normrnd(avg,sigma, hidden_size, hidden_size);
% Wh = eye(hidden_size);
V1 = 0.01 * normrnd(avg,sigma, hidden_size, input_size);
V2 = 0.01 * normrnd(avg,sigma, hidden_size, hidden_size);
Wo = 0.1 * normrnd(avg,sigma, output_size, hidden_size);

Bk = zeros(hidden_size, 1);
Bz = zeros(hidden_size, 1);
Bh = zeros(hidden_size, 1);
Bo = zeros(output_size, 1);

theta = [Wi(:); Wh(:); V1(:); V2(:); Wo(:); Bk(:); Bz(:); Bh(:); Bo(:)]; 


