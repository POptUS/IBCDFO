function [h] = emittance_h(F)

assert(length(F) == 3, "Emittance must have exactly 3 inputs");
h = F(1) * F(2) - F(end)^2;

end
