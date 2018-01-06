function theta = GRU_init(input_size, hidden_size, output_size)
avg = 0;
sigma = 1;

% r = sqrt(6) / sqrt(input_size + hidden_size + output_size); 
% W1 = rand(hidden_size, input_size) * 2 * r - r;
% W2 = rand(hidden_size, hidden_size) * 2 * r - r;
% U1 = rand(hidden_size, input_size) * 2 * r - r;
% U2 = rand(hidden_size, hidden_size) * 2 * r - r;
% V1 = rand(hidden_size, input_size) * 2 * r - r;
% V2 = rand(hidden_size, hidden_size) * 2 * r - r;
% W3 = rand(output_size, hidden_size) * 2 * r - r;

W1 = 0.1 * normrnd(avg,sigma, hidden_size, input_size);
W2 = 0.1 * normrnd(avg,sigma, hidden_size, hidden_size);
U1 = 0.01 * normrnd(avg,sigma, hidden_size, input_size);
U2 = 0.01 * normrnd(avg,sigma, hidden_size, hidden_size);
V1 = 0.01 * normrnd(avg,sigma, hidden_size, input_size);
V2 = 0.01 * normrnd(avg,sigma, hidden_size, hidden_size);
W3 = 0.1 * normrnd(avg,sigma, output_size, hidden_size);

Bk = zeros(hidden_size, 1);
Bz = zeros(hidden_size, 1);
Br = zeros(hidden_size, 1);
Bo = zeros(output_size, 1);
theta = [W1(:); W2(:); U1(:); U2(:); V1(:); V2(:); W3(:); Bk(:); Bz(:); Br(:); Bo(:)]; 


