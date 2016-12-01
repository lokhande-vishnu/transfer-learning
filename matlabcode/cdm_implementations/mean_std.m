function [ mat_out ] = mean_std( mat_in )
mean_d = ones(size(mat_in,1),1)*mean(mat_in,1);
std_d = ones(size(mat_in,1),1)*std(mat_in,1);
mat_out = (mat_in - mean_d) ./ std_d;

% min_d = ones(size(mat_in,1),1)*min(mat_in);
% max_d = ones(size(mat_in,1),1)*max(mat_in);
% mat_out = (mat_in - min_d)./(max_d - min_d);

end

