%% LOAD DATASET
dataset = narma(30, 10000, 1);

X = mapminmax(dataset.inputs);
y = mapminmax(dataset.outputs);
 
% X = dataset.inputs;
% y = dataset.outputs;

X_tr = X(1:4000);
y_tr = y(1:4000);

X_val = X(4001:5000);
y_val = y(4001:5000);

X_ts = X(5001:end);
y_ts = y(5001:end);

% add biases to data
X_tr  = [X_tr; ones(1, size(X_tr, 2))]; 
X_val = [X_val; ones(1, size(X_val, 2))];
X_ts  = [X_ts; ones(1, size(X_ts, 2))];

ni = size(X, 1);

%IP = false; % enable/disable IP training
IP = true;
pname = 'narma_30';

%% PARAM GRID
nrs = [100];           % reservoir units
rhos = [0.7 0.8 0.9 0.99];     % spectral radius
lambdas = [1e-3 1e-4 1e-5 1e-6 0];     % regularization
scalings = [10 1 0.1 0.01];        % input scaling
connectivities = [0.1 0.3 0.5 0.7 0.9]; % # of connected units

if IP
    %IP hyperparams
    etas = [1e-5 1e-4 1e-3];
    mus = [0.25 0 -0.25];
    sigmas = [0.1 0.2 0.5];
else
    etas = 0;
    mus = 0;
    sigmas = 0;
end

% other params
ntransient = 100;
nepoch = 10;
esn_pool = 100;

%% MODEL SELECTION

keep_track_rho   = [];
keep_track_mu = [];
keep_track_sigma = [];

nr_best = 0;
rho_best = 0;
lambda_best = 0;
scaling_best = 0;
connectivity_best = 0;
eta_best = 0;
mu_best = 0;
sigma_best = 0;

error_tr_avg_best = Inf;
error_val_avg_best = Inf;

W_in_best = [];
W_hat_best = [];
W_out_best = [];

fprintf('- begin grid search\n');

for nr = nrs
for rho = rhos
for lambda = lambdas
for scaling = scalings
for connectivity = connectivities
for eta = etas
for mu = mus
for sigma = sigmas
    
    fprintf('- ESN params: nres: %d,\trho: %f,\tlambda: %f,\tscaling: %f,\tconnectivity: %f,\n',...
        nr, rho, lambda, scaling, connectivity); 
    
    fprintf('- IP params: eta: %f,\tmu: %f,\t sigma: %f\n', eta, mu, sigma); 

    
    %% ESN TRAINING
    err_pool_tr = [];
    err_pool_val = [];
        
    for i=1:esn_pool
        % training
        [W_in, W_hat] = echo_state_network(ni, nr, rho, scaling, connectivity);
        
        if IP
            [ip_a, ip_b]  = esn_train_ip(X_tr, W_in, W_hat, eta, mu, sigma, nepoch);
        else
            ip_a = ones(nr, 1);
            ip_b = zeros(nr, 1);
        end
        
        [ W_out, output_tr, err_tr ] = esn_train(X_tr, y_tr, W_in, W_hat, lambda, ip_a, ip_b, ntransient);
        err_pool_tr(end + 1) = err_tr;
        
        % validation
        [ output_val, err_val ] = esn_score(X_val, y_val, W_in, W_hat, W_out, ip_a, ip_b, ntransient);
        err_pool_val(end + 1) = err_val;
    end
    
    % considering the mean of pool
    error_tr_avg  = mean(err_pool_tr);
    error_val_avg = mean(err_pool_val);
    error_tr_std  = std(err_pool_tr);
    error_val_std = std(err_pool_val);
    
    % update keep track info
    keep_track_rho = [keep_track_rho; rho, error_val_avg, error_val_std];
    keep_track_sigma = [keep_track_sigma; sigma, error_val_avg, error_val_std];
    keep_track_mu = [keep_track_mu; mu, error_val_avg, error_val_std];
    
    fprintf('-- TR error: %f (%f),\t - VAL error: %f (%f)\n',...
        error_tr_avg, error_tr_std, error_val_avg, error_val_std);
    
    % check to find a new best
    if error_val_avg < error_val_avg_best
        fprintf('-- FOUND NEW BEST!\n');
        
        %save errors
        error_val_avg_best = error_val_avg;
        error_tr_avg_best = error_tr_avg;
        error_val_std_best = error_val_std;
        error_tr_std_best = error_tr_std;
        
        %save parameters
        nr_best = nr;
        rho_best = rho;
        lambda_best = lambda;
        scaling_best = scaling;
        connectivity_best = connectivity;
        eta_best = eta;
        mu_best = mu;
        sigma_best = sigma;
    end
   
end
end
end
end
end
end
end
end
fprintf('- end grid search\n');
fprintf('- best ESN params: nres: %d,\trho: %f,\tlambda: %f,\tscaling: %f,\tconnectivity: %f,\n',...
    nr_best, rho_best, lambda_best, scaling_best, connectivity_best); 
fprintf('- best IP params: eta: %f,\tmu: %f,\t sigma: %f\n', eta_best, mu_best, sigma_best); 
fprintf('- best TR error: %f (%f),\t - best VAL error: %f (%f)\n',...
        error_tr_avg_best, error_tr_std_best, error_val_avg_best, error_val_std_best);
    
%% TRAIN WITH FULL DATASET (TR+VAL)
fprintf('- retraining model with full dataset\n');

err_pool_tr = [];
err_pool_ts = [];
outputs_ts = [];
y_tr_pred_pool = [];
y_ts_pred_pool = [];

for i = 1:esn_pool
    % training
    [W_in, W_hat] = echo_state_network(ni, nr_best, rho_best, scaling_best, connectivity_best);
    
    if IP
        [ip_a, ip_b]  = esn_train_ip(X_tr, W_in, W_hat, eta_best, mu_best, sigma_best, nepoch);
    else
        ip_a = ones(nr_best, 1);
        ip_b = zeros(nr_best, 1);
    end
    
    [ W_out, y_tr_pred, err_tr ] = esn_train([X_tr X_val], [y_tr y_val], W_in, W_hat, lambda_best, ip_a, ip_b, ntransient);
    
    err_pool_tr(end + 1) = err_tr;
    y_tr_pred_pool = [y_tr_pred_pool; y_tr_pred];
    
    % test
    [ y_ts_pred, err_ts ] = esn_score(X_ts, y_ts, W_in, W_hat, W_out, ip_a, ip_b, ntransient);
    err_pool_ts(end + 1) = err_ts;
    y_ts_pred_pool = [y_ts_pred_pool; y_ts_pred];
end

% considering the mean of pool
error_tr_avg_final  = mean(err_pool_tr);
error_ts_avg_final = mean(err_pool_ts);
error_tr_std_final  = std(err_pool_tr);
error_ts_std_final = std(err_pool_ts);

y_tr_pred_avg = mean(y_tr_pred_pool, 1);
y_ts_pred_avg = mean(y_ts_pred_pool, 1);
y_tr_pred_std = std(y_tr_pred_pool, 1);
y_ts_pred_std = std(y_ts_pred_pool, 1);

fprintf('\n*** %s ***\n', pname);
fprintf('- final TR error: %f (%f),\t - final TS error: %f (%f)\n',...
    error_tr_avg_final, error_tr_std_final, error_ts_avg_final, error_ts_std_final);

%% PLOT
% target vs output
figure
subplot(2, 1, 1);
hold on
y_temp = [y_tr y_val];
plot(1:size(y_tr_pred_avg, 2), y_temp(:,ntransient:end));
plot(1:size(y_tr_pred_avg, 2), y_tr_pred_avg);
plot(1:size(y_tr_pred_avg, 2), y_tr_pred_avg + y_tr_pred_std, '--');
plot(1:size(y_tr_pred_avg, 2), y_tr_pred_avg - y_tr_pred_std, '--');

title('target vs output (TR+VAL)');
legend('target', 'output', 'upper bound', 'lower bound');


subplot(2, 1, 2);
hold on
plot(1:size(y_ts_pred_avg, 2), y_ts(:,ntransient:end));
plot(1:size(y_ts_pred_avg, 2), y_ts_pred_avg);
plot(1:size(y_ts_pred_avg, 2), y_ts_pred_avg + y_ts_pred_std, '--');
plot(1:size(y_ts_pred_avg, 2), y_ts_pred_avg - y_ts_pred_std, '--');
title('target vs output (TS)');
legend('target', 'output', 'upper bound', 'lower bound');

print(strcat(pname, '_esn_output_target'),'-dpng');
savefig(strcat(pname, '_esn_output_target'));

figure
hold on
plot(keep_track_rho(:, 1), keep_track_rho(:, 2));
plot(keep_track_rho(:, 1), keep_track_rho(:, 2) + keep_track_rho(:, 3), '--');
plot(keep_track_rho(:, 1), keep_track_rho(:, 2) - keep_track_rho(:, 3), '--');
title('performance w.r.t. rho');
legend('mean', 'upper bound', 'lower bound');

print(strcat(pname, '_esn_output_target'),'-dpng');
savefig(strcat(pname, '_esn_output_target'));

figure
hold on
plot(keep_track_sigma(:, 1), keep_track_sigma(:, 2));
plot(keep_track_sigma(:, 1), keep_track_sigma(:, 2) + keep_track_sigma(:, 3), '--');
plot(keep_track_sigma(:, 1), keep_track_sigma(:, 2) - keep_track_sigma(:, 3), '--');
title('performance w.r.t. sigma');
legend('mean', 'upper bound', 'lower bound');

print(strcat(pname, '_keep_track_sigma'),'-dpng');
savefig(strcat(pname, '_keep_track_sigma'));

figure
hold on
plot(keep_track_mu(:, 1), keep_track_mu(:, 2));
plot(keep_track_mu(:, 1), keep_track_mu(:, 2) + keep_track_mu(:, 3), '--');
plot(keep_track_mu(:, 1), keep_track_mu(:, 2) - keep_track_mu(:, 3), '--');
title(strcat(pname,'performance w.r.t. mu'));
legend('mean', 'upper bound', 'lower bound');

print(strcat(pname, '_keep_track_mu'),'-dpng');
savefig(strcat(pname, '_keep_track_mu'));
n
plot(keep_track_mu(:, 1), keep_track_mu(:, 2));
plot(keep_track_mu(:, 1), keep_track_mu(:, 2) + keep_track_mu(:, 3), '--');
plot(keep_track_mu(:, 1), keep_track_mu(:, 2) - keep_track_mu(:, 3), '--');
title(strcat(pname,'performance w.r.t. mu'));
legend('mean', 'upper bound', 'lower bound');

print(strcat(pname, '_keep_track_mu'),'-dpng');
savefig(strcat(pname, '_keep_track_mu'));
