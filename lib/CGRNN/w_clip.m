function w_grad = w_clip(w_grad, w, lamda, gc)
if sqrt(sum(sum(w_grad.^2))) > gc
    w_grad = gc * w_grad / sqrt(sum(sum(w_grad.^2))) + lamda * w;
else
    w_grad = w_grad + lamda * w;
end