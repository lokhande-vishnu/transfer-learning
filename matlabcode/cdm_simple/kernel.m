function [ k_out ] = kernel( k_in1, k_in2,c,c2 )
%     k_out = c*exp(-(k_in1*k_in2')./(2*c2));
    for i = 1:size(k_in1,1)
        for j = 1:size(k_in2,1)
            k_out(i,j) = norm(k_in1(i,:) - k_in2(j,:),2)^2;
        end
    end

end
