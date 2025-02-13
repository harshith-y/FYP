import deepxde as dde
import tensorflow as tf
import numpy as np

def finite_difference_derivative(f, t, delta_t):
    """
    Compute the first derivative using finite differences.
    :param f: Function values (Tensor).
    :param t: Time values (Tensor).
    :param delta_t: Step size for finite differences.
    :return: First derivative (Tensor).
    """
    t_plus = t + delta_t
    t_minus = t - delta_t
    f_plus = tf.convert_to_tensor(f(t_plus), dtype=tf.float32)
    f_minus = tf.convert_to_tensor(f(t_minus), dtype=tf.float32)
    return (f_plus - f_minus) / (2 * delta_t)


def finite_difference_second_derivative(f, t, delta_t):
    """
    Compute the second derivative using finite differences.
    :param f: Function values (Tensor).
    :param t: Time values (Tensor).
    :param delta_t: Step size for finite differences.
    :return: Second derivative (Tensor).
    """
    t_plus = t + delta_t
    t_minus = t - delta_t
    f_plus = tf.convert_to_tensor(f(t_plus), dtype=tf.float32)
    f_minus = tf.convert_to_tensor(f(t_minus), dtype=tf.float32)
    f_center = tf.convert_to_tensor(f(t), dtype=tf.float32)
    return (f_plus - 2 * f_center + f_minus) / (delta_t ** 2)

class GlucoseModel:
    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.parameters = {
            2: {'M': 72, 'ksi': 197, 'kl': 1.94, 'Tu': 122, 'ku_Vi': 59e-3, 'Tr': 183, 'kr_Vb': 2.4e-3, 'tend': 48 * 60},
            3: {'M': 94, 'ksi': 274, 'kl': 1.72, 'Tu': 88, 'ku_Vi': 62e-3, 'Tr': 49, 'kr_Vb': 2e-3, 'tend': 48 * 60},
            # Add other patient parameters...
        }
        self.set_parameters(patient_id)
        self.params_to_inverse()
        self.normalize_parameters()

        # Time domain for simulation
        self.min_t = 0
        self.max_t = self.parameters[patient_id]['tend']

    def set_parameters(self, patient_id):
        if patient_id in self.parameters:
            params = self.parameters[patient_id]
            self.M = params['M']
            self.kl = params['kl']
            self.kb = 128 / self.M
            self.Tu = params['Tu']
            self.Tr = params['Tr']
            self.Vi = 2.5 * self.M
            self.Vb = 0.65 * self.M
            self.ksi = params['ksi']
            self.insulin_times = [450, 750, 1020, 1440, 2220, 2250, 2550, 2880]
            self.insulin_values = [0.5, 2.0, 2.0, 22.0, 18.0, 17.0, 16.0, 19.0]
            self.carb_times = [1440, 1530, 2220, 2460, 2550, 2670]
            self.carb_values = [128, 15, 150, 100, 7.5, 15]
        else:
            raise ValueError("Patient ID not recognized.")

    def params_to_inverse(self):
        self.ksi = tf.Variable(self.ksi, trainable=True, dtype=tf.float32)

    def normalize_parameters(self):
        self.G_mean, self.G_std = 110, 40
        self.I_mean, self.I_std = 0.5, 0.1
        self.D_mean, self.D_std = 100, 50
        self.normalized_kl = self.kl / self.G_std
        self.normalized_kb = self.kb / self.G_std
        self.normalized_ksi = self.ksi / self.G_std

    def geometry_time(self):
        return dde.geometry.TimeDomain(self.min_t, self.max_t)

    def time_inputs(self, t):
        """
        Define time-dependent insulin and carbohydrate inputs.
        :param t: Time variable (1D tensor).
        :return: Insulin and carbohydrate input values at time t.
        """
        # Reshape t for broadcasting
        t = tf.reshape(t, [-1, 1])  # Reshape to [batch_size, 1]

        # Broadcast comparison
        U_conditions = tf.equal(t, self.insulin_times)  # Compare all t with insulin_times
        R_conditions = tf.equal(t, self.carb_times)    # Compare all t with carb_times

        # Compute insulin and carbohydrate inputs
        U = tf.reduce_sum(tf.where(U_conditions, self.insulin_values, 0.0), axis=1)
        R = tf.reduce_sum(tf.where(R_conditions, self.carb_values, 0.0), axis=1)

        return U, R


    def ode(self, t, y):
        """
        Define the ODE system for glucose, insulin, and carbohydrate digestion dynamics.
        :param t: Time variable (1D tensor).
        :param y: State variables [G, I, D].
        :return: Residuals for glucose, insulin, and carbohydrate equations.
        """
        delta_t = 1e-3  # Step size for finite differences

        # Define state variables as functions of t
        G = lambda t_val: y[:, 0:1] + 0.1 * t_val  # Example: G explicitly depends on t
        I = lambda t_val: y[:, 1:2] + 0.1 * t_val
        D = lambda t_val: y[:, 2:] + 0.1 * t_val

        # Compute first and second derivatives using finite differences
        dG_dt = finite_difference_derivative(G, t, delta_t)
        dI_dt = finite_difference_derivative(I, t, delta_t)
        d2I_dt2 = finite_difference_second_derivative(I, t, delta_t)
        dD_dt = finite_difference_derivative(D, t, delta_t)
        d2D_dt2 = finite_difference_second_derivative(D, t, delta_t)

        # Residual equations
        eq_1 = (dG_dt - (-self.ksi / self.G_std * I(t)) + D(t) + self.normalized_kl - self.normalized_kb) / self.G_std
        eq_2 = d2I_dt2 - ((-1 / self.Tu**2) * I(t)) + ((-2 / self.Tu) * dI_dt) + (
            (1 / self.Vi * (1 / self.Tu**2))
        )
        eq_3 = d2D_dt2 - ((-1 / self.Tr**2) * D(t)) + ((-2 / self.Tr) * dD_dt) + (
            (1 / self.Vb * (1 / self.Tr**2))
        )

        return [eq_1, eq_2, eq_3]









    def compute_loss(self, observed_glucose=None, time_points=None):
        time = np.linspace(self.min_t, self.max_t, 100)
        time = tf.convert_to_tensor(time, dtype=tf.float32)
        initial_conditions = tf.random.uniform([100, 3], minval=0, maxval=1, dtype=tf.float32)
        residuals = self.ode(time, initial_conditions)
        residual_loss = tf.reduce_mean([tf.reduce_mean(tf.square(res)) for res in residuals])

        data_loss = 0
        if observed_glucose is not None and time_points is not None:
            predicted_glucose = self.predict_glucose(time_points)
            data_loss = tf.reduce_mean(tf.square(predicted_glucose - observed_glucose))

        total_loss = residual_loss + data_loss
        return total_loss
