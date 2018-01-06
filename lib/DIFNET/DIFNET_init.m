function theta = DIFNET_init(input_size, hidden_size, output_size)
avg = 0;
sigma = 1;

Wp = 0.1 * normrnd(avg,sigma, input_size, hidden_size);
Wk = 0.1 * normrnd(avg,sigma, hidden_size, input_size);
Wh = 0.1 * normrnd(avg,sigma, hidden_size, hidden_size);
% Wh = 1 * eye(hidden_size);
Wz = 0.1 * normrnd(avg,sigma, hidden_size, input_size);
Wo = 0.1 * normrnd(avg,sigma, output_size, hidden_size);
Bp = zeros(input_size, 1);
Bk = zeros(hidden_size, 1);
Bz = zeros(hidden_size, 1);
Bo = zeros(output_size, 1);

theta = [Wp(:); Wk(:); Wh(:); Wz(:); Wo(:); Bp(:); Bk(:); Bz(:); Bo(:)];

