%% TRAIN THE PARAMETERS OF THE IP LEARNING RULE
function [ ip_a, ip_b ] = esn_train_ip(X, W_in, W_hat, eta, mu, sigma, nepochs)
    % X: samples on the columns
    tol = 1e-3;
    % init IP params
    ip_a = ones(size(W_hat, 1), 1);
    ip_b = zeros(size(W_hat, 1), 1);

    for epoch = 1:nepochs
        old_a = ip_a;
        old_b = ip_b;
        
        for j = 1:size(X, 2)
            if j == 1
                x = W_in * X(:, j);
            else
                x = W_in * X(:, j) + W_hat * last_state;
            end

            %save last echo states (needed for x(t+1))
            y = tanh(ip_a .* x + ip_b);
            last_state = y;

            % fprintf('epoch %d,\ta: [%s],\tb: [%s] \n', j, sprintf('%f ', ip_a), sprintf('%f ', ip_b));

            % IP delta rule
            zs = zeros(size(W_hat, 1), 1);
            delta_b = - eta .* ( - (zs + mu./sigma.^2) +...
                              y./sigma.^2 .* (2.*sigma.^2 + 1 - y.^2 + mu.*y) );
            ip_b = ip_b + delta_b;
            delta_a = eta ./ ip_a + delta_b .* x;
            ip_a = ip_a + delta_a;
        end
        
         % stop if change is negligible
         if norm(old_a - ip_a) < tol && norm(old_b - ip_b) < tol
                 break
         end
    end
end