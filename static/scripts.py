from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.integrate import cumtrapz
import rrcf
import pywt
from pykalman import KalmanFilter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
# from itertools import cycle
# from scipy.stats import zscore
import numpy as np
import pandas as pd
# import io
# import csv
from numpy.fft import fft, fftfreq


# def process_csv(file_path):
#     df = pd.read_csv(file_path)
#     class_counts = df['CLASS_NAME'].value_counts().to_dict()
#     samples = []
#
#     grouped = df.groupby('SAMPLE_INDEX')
#     for name, group in grouped:
#         sample = {
#             'sampleIndex': name,
#             'class': group['CLASS_NAME'].iloc[0],
#             'data': group.drop(columns=['SAMPLE_INDEX', 'CLASS_NAME']).to_dict(orient='records')
#         }
#         samples.append(sample)
#
#     print(class_counts)
#     return samples, class_counts


def process_csv(file_path):
    df = pd.read_csv(file_path)

    # print(group.columns[group.columns.str.contains('Unnamed')])
    col2drop = df.columns[df.columns.str.contains('Unnamed')].tolist()
    df.drop(columns=col2drop, inplace=True)

    samples = []
    class_counts = {}

    # Group by sample_index and collect data for each sample
    grouped = df.groupby('SAMPLE_INDEX')
    for name, group in grouped:
        sample_class = group['CLASS_NAME'].iloc[0]
        if sample_class not in class_counts:
            class_counts[sample_class] = 0
        class_counts[sample_class] += 1

        sample = {
            'sampleIndex': name,
            'class': sample_class,
            'data': group.drop(columns=['SAMPLE_INDEX', 'CLASS_NAME']).to_dict(orient='records')
        }
        samples.append(sample)

    return samples, class_counts


def convert_numerical_strings(params):
    """
    Converts numerical string values in a dictionary to integers or floats.

    Args:
    params (dict): Dictionary containing parameters with potential numerical string values.

    Returns:
    dict: Dictionary with numerical string values converted to int or float.
    """

    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def is_int(value):
        try:
            int(value)
            return True
        except ValueError:
            return False

    converted_params = {}

    for key, value in params.items():
        if isinstance(value, str):
            if is_int(value):
                converted_params[key] = int(value)
            elif is_float(value):
                converted_params[key] = float(value)
            else:
                converted_params[key] = value
        else:
            converted_params[key] = value

    return converted_params

def smooth(data, window_size=5):
    if window_size < 2:
        return data
    cumsum = [0] + list(np.cumsum(data))
    smoothed = [(cumsum[i] - cumsum[max(0, i - window_size)]) / min(i, window_size) for i in range(1, len(cumsum))]
    return smoothed


def exponential_smoothing(data, alpha=0.3):
    # Exponential smoothing formula
    result = [data[0]]  # first value is same as series
    for n in range(1, len(data)):
        result.append(alpha * data[n] + (1 - alpha) * result[n-1])
    return result


def zscore_anomaly(values):

    # Calculate rolling mean and standard deviation
    window = 20
    if len(values) < window:
        return -1

    padded_values = np.pad(values, (window - 1, 0), mode='edge')
    rolling_mean = np.convolve(padded_values, np.ones(window) / window, mode='valid')
    rolling_var = np.convolve(padded_values ** 2, np.ones(window) / window, mode='valid') - rolling_mean ** 2
    rolling_std = np.sqrt(np.maximum(rolling_var, 0))  # Ensure non-negative argument for sqrt

    # rolling_std = np.sqrt(
    #     np.convolve(padded_values ** 2, np.ones(window) / window, mode='valid') - rolling_mean ** 2
    # )

    # Identify anomalies
    anomalies = np.abs(values - rolling_mean) > 3 * rolling_std
    return anomalies


def isolation_forest_anomaly(values, contamination='auto'):
    num_samples = len(values)
    if num_samples < 2:
        return np.array([0] * num_samples)

    values = np.array(values).reshape(-1, 1)
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(values)

    model = IsolationForest(contamination=contamination, n_estimators=100, max_samples=num_samples)
    predictions = model.fit_predict(scaled_values)

    return (predictions == -1).astype(int)  # Return 1 for anomalies, 0 for normal points


def rrcf_anomaly(values, num_trees=40, tree_size=256, shingle_size=2, threshold=None):
    if len(values) < shingle_size or len(values) < 2:
        return np.zeros(len(values), dtype=int)

    # Shingle the data
    points = rrcf.shingle(values, size=shingle_size)
    points = np.vstack([point for point in points])
    n = points.shape[0]

    # Initialize the forest
    forest = []
    for _ in range(num_trees):
        tree = rrcf.RCTree()
        forest.append(tree)

    # Anomaly scores
    anomaly_scores = np.zeros(n)

    # For each shingle...
    for i in range(n):
        point = tuple(points[i, :])

        # For each tree in the forest...
        for tree in forest:
            if len(tree.leaves) > tree_size:
                tree.forget_point(i - tree_size)
            tree.insert_point(point, index=i)
            anomaly_scores[i] += tree.codisp(i)

    # Normalize anomaly scores
    anomaly_scores /= num_trees

    # If a threshold is not provided, set it to a default value
    if threshold is None:
        threshold = 1.5 * np.median(anomaly_scores)

    # Return a binary anomaly label
    anomalies = (anomaly_scores > threshold).astype(int)
    return np.concatenate([np.zeros(shingle_size - 1, dtype=int), anomalies])


def spectral_residual_transform(data, window_size=5):

    freq = np.fft.fft(data)
    mag = np.sqrt(freq.real ** 2 + freq.imag ** 2)
    spectral_residual = np.exp(np.log(mag + 1e-8) - series_filter(np.log(mag + 1e-8), window_size))

    freq.real = freq.real * spectral_residual / mag
    freq.imag = freq.imag * spectral_residual / mag

    saliency_map = np.fft.ifft(freq)

    spectral_residual = np.sqrt(saliency_map.real ** 2 + saliency_map.imag ** 2)
    return spectral_residual.tolist()


def series_filter(values, kernel_size=3):
    """
    Filter a time series. Practically, calculated mean value inside kernel size.
    As math formula, see https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html.
    :param values:
    :param kernel_size:
    :return: The list of filtered average
    """
    filter_values = np.cumsum(values, dtype=float)

    filter_values[kernel_size:] = filter_values[kernel_size:] - filter_values[:-kernel_size]
    filter_values[kernel_size:] = filter_values[kernel_size:] / kernel_size

    for i in range(1, kernel_size):
        filter_values[i] /= i + 1

    return filter_values

def derivative_transform(data):
    derivative = np.gradient(data)
    return derivative.tolist()


def integral_transform(data):
    # Compute the cumulative integral using the trapezoidal rule
    integral = cumtrapz(data, initial=0)
    return integral.tolist()


def logarithm_transform(data):
    print(data)
    log_data = np.log(data)
    return log_data.tolist()


def fourier_transform(data):
    # Compute the Fourier Transform
    fft_result = np.fft.fft(data)
    # Compute the magnitudes (absolute values)
    magnitudes = np.abs(fft_result)
    # Only return half of the FFT result (since it's symmetric for real inputs)
    return magnitudes[:len(data) // 2].tolist()


def perform_fft(data, sampling_rate=1):
    # Perform FFT
    fft_result = fft(data)
    n = len(data)
    freq = fftfreq(n, d=1/sampling_rate)

    # Extracting half of the frequency and corresponding power
    half_n = n // 2
    freq = freq[:half_n]
    fft_power = 2.0 / n * np.abs(fft_result[:half_n])
    return freq.tolist(), fft_power.tolist()


def wavelet_transform(data, wavelet='db1', level=1, coefficient='approximation'):
    # Apply discrete wavelet transform
    coeffs = pywt.wavedec(data, wavelet, level=level)
    # Select the appropriate coefficients
    if coefficient == 'approximation':
        return coeffs[0].tolist()  # Approximation coefficients
    elif coefficient == 'detail':
        # Concatenate all detail coefficients
        details = np.concatenate(coeffs[1:])
        return details.tolist()
    else:
        raise ValueError("Invalid coefficient type")


def calculate_dynamic_moments_optimized(data, moment_index = 1):
    Y = np.array(data) # Rename for clarity
    Y_derivative = np.array(derivative_transform(data))

    length = len(Y)
    inv = 1.0 / (np.arange(length) + 1)  # Avoiding division by zero

    Y_pow_2 = Y ** 2
    Y_pow_3 = Y ** 3
    Y_deriv_pow_2 = Y_derivative ** 2

    if moment_index == 0:  # MD2
        MD2 = np.cumsum(Y * Y_derivative) * inv
        return MD2.tolist()
    elif moment_index == 1:  # MD3x
        MD3x = (inv / 2) * (Y_pow_3 - 3 * Y * Y_deriv_pow_2)
        return MD3x.tolist()
    elif moment_index == 2:  # MD3y
        MD3y =  (inv / 2) * (Y_pow_3 - 3 * Y_pow_2 * Y_derivative)
        return MD3y.tolist()
    elif moment_index == 3:  # MD3pb
        MD3pb = (np.sqrt(2) * inv / 2) * (Y_pow_2 * Y_derivative - Y * Y_deriv_pow_2)
        return MD3pb.tolist()
    elif moment_index == 4:  # MD3sb
        MD3sb = (np.sqrt(2) * inv / 2) * (2 * Y_pow_3 + 3 * (Y_pow_2 * Y_derivative - Y * Y_deriv_pow_2))
        return MD3sb.tolist()
    else:
        raise ValueError("Invalid moment_index. Choose a value between 0 and 4.")


def autocorrelation(data):
    n = len(data)
    data_mean = np.mean(data)
    autocorr = np.correlate(data - data_mean, data - data_mean, mode='full') / (n * np.var(data))
    return autocorr[n-1:].tolist()  # Return only the second half


def cross_correlation(data1, data2):
    if len(data1) != len(data2):
        raise ValueError("Data series must have the same length for cross-correlation")

    n = len(data1)
    mean1, mean2 = np.mean(data1), np.mean(data2)
    crosscorr = np.correlate(data1 - mean1, data2 - mean2, mode='full') / (n * np.std(data1) * np.std(data2))
    return crosscorr[n-1:].tolist()  # Return only the second half


def apply_kalman_filter_to_multiple_sensors(*sensor_values):
    # Number of sensors
    num_sensors = len(sensor_values)

    # Check if all sensors have the same number of data points
    if not all(len(values) == len(sensor_values[0]) for values in sensor_values):
        raise ValueError("All sensors must have the same number of data points")

    # Convert the sensor data into a numpy array
    measurements = np.vstack(sensor_values).T

    # Define initial state and covariance
    initial_state = measurements[0]
    initial_state_covariance = 1.0

    # Define observation and transition matrices
    observation_matrix = np.eye(1)
    transition_matrix = np.eye(1)

    # Define process and observation noise
    process_noise = 0.05  # Tune this based on your data
    observation_noise = 1.0  # Tune this based on your data

    print(observation_noise)
    kf = KalmanFilter(
        initial_state_mean=np.zeros(num_sensors),
        n_dim_obs=num_sensors,
        n_dim_state=num_sensors,  # Assuming state and observation dimensions are the same
        initial_state_covariance=initial_state_covariance,
        observation_matrices=observation_matrix,
        transition_matrices=transition_matrix,
        observation_covariance=observation_noise,
        transition_covariance=process_noise
    )

    print(kf)

    # Apply the Kalman Filter
    state_means, _ = kf.filter(measurements)
    return state_means


def normalise_data(data, method='standard'):
    """
    Transforms the input data using the specified method.

    Parameters:
    - data (numpy.ndarray): The input data array of shape [features, sequence length].
    - method (str): The method of transformation. Options include:
        'standard' - Standard Scaling
        'minmax' - MinMax Scaling
        'maxabs' - MaxAbs Scaling
        'robust' - Robust Scaling
        'log' - Logarithmic Transformation
        'yeojohnson' - Yeo-Johnson Power Transformation
        'boxcox' - Box-Cox Power Transformation (requires all positive data)

    Returns:
    - numpy.ndarray: The transformed data array.
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'maxabs':
        scaler = MaxAbsScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'log':
        # Apply log transformation using log1p to deal with zero values safely
        return np.log1p(data).tolist()
    elif method == 'yeojohnson':
        scaler = PowerTransformer(method='yeo-johnson')
    elif method == 'boxcox':
        # Ensure data is positive before applying Box-Cox
        if np.any(data <= 0):
            raise ValueError("Box-Cox transformation requires all positive data.")
        scaler = PowerTransformer(method='box-cox')
    else:
        raise ValueError(f"Unknown transformation method: {method}")

    # Fit and transform the data
    # Reshape data for fitting if it's 2D already, assumed to be [features, sequence length]
    # if data.ndim == 2:
    #     data = data.reshape(-1, 1)  # Treat each feature-time point as independent

    data = data.reshape(-1, 1)

    transformed_data = scaler.fit_transform(data)

    transformed_data = transformed_data.reshape(-1)

    # # Reshape back to original shape if it was 2D
    # if data.ndim == 2:
    #     transformed_data = transformed_data.reshape(-1, data.shape[1])

    return transformed_data.tolist()


def generate_polynomial_features(data, degree=2, include_bias=False):
    """
    Generates polynomial and interaction features for the given data.

    Parameters:
    - data (numpy.ndarray): The input data array of shape [1, features, sequence length].
    - degree (int): The degree of the polynomial features. Default is 2.
    - include_bias (bool): If True, includes a bias column (column of ones) as feature.

    Returns:
    - numpy.ndarray: An array of shape [features x expanded sequence length] containing
      the original and polynomial features.

    Explanation and Customization:
    Handling Data Format: This function assumes each column of your input data (data) is a separate time point with
    measurements from one sensor. It applies PolynomialFeatures to the data across the time sequence.
    Parameters:
        degree: Controls the complexity of the polynomial features (e.g., quadratic, cubic).
        include_bias: Whether to include a bias column (all ones), which can act as an intercept in many linear models.
    Output: The output is an expanded set of features that now includes the polynomial terms, reshaped to keep the time
    sequence intact.
    """
    # Flatten the data array as it has an extra dimension with only one feature
    flattened_data = data.reshape(-1, 1)

    # Instantiate the PolynomialFeatures object
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)

    # Apply PolynomialFeatures to the flattened data
    transformed_data = poly.fit_transform(flattened_data)

    return transformed_data.tolist()


def decompose_time_series(data, model='additive', output_key='resid', freq=None):
    """
    Decomposes the time series data into trend, seasonal, and residual components.

    Parameters:
    - data (numpy.ndarray): The input data array of shape [features, sequence length].
    - model (str): Type of decomposition model ('additive' or 'multiplicative').
    - freq (int): The frequency of the cycle in the time series (e.g., if a seasonal
                  cycle is annual and the data is monthly, freq would be 12).

    Returns:
    - dict: A dictionary containing the decomposed components.
    """
    decompositions = {}


    series = data
    result = sm.tsa.seasonal_decompose(series, model=model, period=freq)
    decompositions = {
        'trend': result.trend,
        'seasonal': result.seasonal,
        'resid': result.resid
    }

    selected_decomposition = decompositions[output_key]#{key: value[output_key] for key, value in decompositions.items()}

    # Convert the dictionary of residuals into a DataFrame
    selected_decomposition = np.array(selected_decomposition)

    return selected_decomposition.tolist()
