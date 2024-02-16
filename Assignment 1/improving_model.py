from main import run

ksize_first = [3, 5, 7]
ksize_second = [3, 5, 7]
sigma_x = [0, 0.5, 0.7]
best_error = None
best_kernel_params = [(0,0),0]

for ksize_f in ksize_first:
    for ksize_s in ksize_second:
        for sigma in sigma_x:
            kernel_params = [(ksize_f, ksize_s), sigma]
            error_i = run(0,True,kernel_params)
            print(f"Model {kernel_params}, error: {error_i}")
            if best_error is None or error_i < best_error:
                best_kernel_params = kernel_params
                best_error = error_i

print(f"The best kernel parameters are: {best_kernel_params} with a total error of {best_error}")