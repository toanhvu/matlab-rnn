function b_grad = b_clip(b_grad, gc)
if sqrt(sum(sum(b_grad.^2))) > gc
    b_grad = gc * b_grad / sqrt(sum(sum(b_grad.^2)));    
end