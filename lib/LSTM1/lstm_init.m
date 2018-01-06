function theta = lstm_init(input_size, hidden_size, output_size)
avg = 0;
sigma = 1;

Wxc = 0.1 * normrnd(avg,sigma, hidden_size, input_size);
Whc = 0.1 * normrnd(avg,sigma, hidden_size, hidden_size);
Bc = zeros(hidden_size, 1);
theta_x = [Wxc(:); Whc(:); Bc(:)];

Wxi = 0.01 * normrnd(avg,sigma, hidden_size, input_size);
Whi = 0.01 * normrnd(avg,sigma, hidden_size, hidden_size);
% Wci = normrnd(avg,sigma, hidden_size, hidden_size);
Wci = 0.01 * diag(normrnd(avg,sigma, hidden_size, 1));
Bi = zeros(hidden_size, 1);
theta_i = [Wxi(:); Whi(:); Wci(:); Bi(:)];

Wxf = 0.01 * normrnd(avg,sigma, hidden_size, input_size);
Whf = 0.01 * normrnd(avg,sigma, hidden_size, hidden_size);
% Wcf = normrnd(avg,sigma, hidden_size, hidden_size);
Wcf = 0.01 * diag(normrnd(avg,sigma, hidden_size, 1));
Bf = zeros(hidden_size, 1);
theta_f = [Wxf(:); Whf(:); Wcf(:); Bf(:)];

Wxo = 0.01 * normrnd(avg,sigma, hidden_size, input_size);
Who = 0.01 * normrnd(avg,sigma, hidden_size, hidden_size);
% Wco = normrnd(avg,sigma, hidden_size, hidden_size);
Wco = 0.01 * diag(normrnd(avg,sigma, hidden_size, 1));
Bo = zeros(hidden_size, 1);
theta_o = [Wxo(:); Who(:); Wco(:); Bo(:)];

W0 = 0.1 * normrnd(avg,sigma, output_size, hidden_size);
B0 = zeros(output_size, 1);
theta0 = [W0(:); B0(:)];

theta = struct;
theta.theta0 = theta0;
theta.theta_x = theta_x;
theta.theta_i = theta_i;
theta.theta_f = theta_f;
theta.theta_o = theta_o;
