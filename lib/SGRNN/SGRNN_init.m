function theta = SGRNN_init(input_size, hidden_size, output_size)
avg = 0;
sigma = 1;

% W1 = 0.01 * normrnd(avg,sigma, hidden_size, input_size);
% W2 = 0.01 * normrnd(avg,sigma, hidden_size, hidden_size);
W1 = 0.04 * rand(hidden_size, input_size) -0.02;
W2 = 0.04 * rand(hidden_size, hidden_size) -0.02;
%W2 = 0.1 * eye(hidden_size);
Bk = zeros(hidden_size, 1);
Bh = zeros(hidden_size, 1);
theta1 = [W1(:); W2(:); Bk(:); Bh(:)];

% Wo = 0.01 * normrnd(avg,sigma, output_size, hidden_size);
Wo = 0.04 * rand(output_size, hidden_size) -0.02;
Bo = zeros(output_size, 1);
theta0 = [Wo(:); Bo(:)];

a = zeros(hidden_size, 1) +0.1; % + 0.1;
b1 = zeros(hidden_size, 1) + 0.8; % + 0.2;
b2 = zeros(hidden_size, 1) + 0.2; %+ 0.8;
theta2 = [a(:); b1(:); b2(:)];

theta.theta1 = theta1;
theta.theta2 = theta2;
theta.theta0 = theta0;




