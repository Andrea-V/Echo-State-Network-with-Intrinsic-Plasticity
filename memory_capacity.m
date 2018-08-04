function [ mc ] = memory_capacity(y_pred, y, ntransient) 
    % y and y_pred: (nout x timeseries lenght) 
    y = y(:, ntransient:end);
    
    mc = 0;
    for s = 2:size(y_pred, 1)
        num = cov(y_pred(s, :), y(s, :)); % <- cov() returns covariane matrix
                                          % [ cov(a, a) cov(b, a) ]
                                          % [ cov(a, b) cov(b, b) ]
        mc_s = num(2, 1).^2 / (var(y_pred(s, :))*var(y(1, :)));
        mc = mc + mc_s;
    end
end