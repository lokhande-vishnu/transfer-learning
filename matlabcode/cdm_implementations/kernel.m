function [ k_out ] = kernel( k_in1, k_in2,c,c2 )
    k_out = c*exp(-(k_in1*k_in2')./(2*c2));
end
