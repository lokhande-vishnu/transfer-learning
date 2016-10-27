function [ k_out ] = kernel( k_in1, k_in2,c,c2 )
    k_out = c*exp(-c2*k_in1*k_in2');
end
