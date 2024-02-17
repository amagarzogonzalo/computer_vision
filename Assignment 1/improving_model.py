from main import run
import numpy as np

# This script aims to provide an hiperparameter tuning for optimizing the kernel used in the preprocessed image.
# For this purpose, several kernel sizes are attempted as well as the sigma x in order to find the optimal combination

def parameters_canny():
    """
    Optimize canny thresholds for image processing.

    This function iterates over different combinations of canny's thresholds 
    to find the best parameters for a specific image processing task.

    :return: The best thresholds and the corresponding error.
    """
    threshold_first = np.linspace(300,600, 5)
    threshold_second = np.linspace(300,600, 5)
    best_error = None
    best_canny_params = (0,0)
    kernel_params = [(3,3),0.5]


    for threshold_f in threshold_first:
        for threshold_s in threshold_second:
            canny_params = (threshold_f, threshold_s)
            error_i = run(0,True,kernel_params, canny_params)
            print(f"Model {kernel_params}, error: {error_i}")
            if error_i != 0 and (best_error is None or error_i < best_error):
                best_canny_params = canny_params
                best_error = error_i

    print(f"The best canny threshold is: {best_canny_params} with a total error of {best_error}")



def kernel_parameters():
    """
    Optimize kernel parameters for image processing.

    This function iterates over different combinations of kernel sizes and sigma values 
    to find the best parameters for a specific image processing task.

    :return: The best kernel parameters and the corresponding error.
    """
    ksize_first = [3, 5, 7]
    ksize_second = [3, 5, 7]
    sigma_x = [0, 0.5, 0.7]
    best_error = None
    best_kernel_params = [(0,0),0]

    for ksize_f in ksize_first:
        for ksize_s in ksize_second:
            for sigma in sigma_x:
                kernel_params = [(ksize_f, ksize_s), sigma]
                error_i = run(2,True,kernel_params, (0,0))
                print(f"Model {kernel_params}, error: {error_i}")
                if best_error is None or error_i < best_error:
                    best_kernel_params = kernel_params
                    best_error = error_i

    print(f"The best kernel parameters are: {best_kernel_params} with a total error of {best_error}")


kernel_parameters()
