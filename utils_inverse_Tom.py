import deepxde as dde
import tensorflow as tf
import numpy as np


class GlucoseModel:
    def __init__(self, pat):
        # Set the parameters based on the patient
        self.set_parameters(pat)
        self.params_to_inverse(pat)

        # Geometry Parameters
        self.min_t = 0
        self.max_t = self.tend  # max time from the patient parameters

        # Initialize insulin and carbohydrate mappings
        self.init_event_mappings(pat)

        # Set scaling factor for glucose residual
        self.set_glucose_scaling(pat)

    def set_parameters(self, pat):
        # Define parameters for all 10 patients
        patient_params = {
            2: {"M": 72, "ksi": 197, "kl": 1.94, "Tu": 122, "ku_Vi": 59e-3, "Tr": 183, "kr_Vb": 2.4e-3, "tend": 48 * 60},
            3: {"M": 94, "ksi": 274, "kl": 1.72, "Tu": 88, "ku_Vi": 62e-3, "Tr": 49, "kr_Vb": 2e-3, "tend": 48 * 60},
            4: {"M": 74, "ksi": 191, "kl": 1.94, "Tu": 126, "ku_Vi": 61e-3, "Tr": 188, "kr_Vb": 2.47e-3, "tend": 48 * 60},
            5: {"M": 91, "ksi": 282, "kl": 1.67, "Tu": 85, "ku_Vi": 64e-3, "Tr": 48, "kr_Vb": 2.06e-3, "tend": 48 * 60},
            6: {"M": 70, "ksi": 203, "kl": 1.94, "Tu": 118, "ku_Vi": 57e-3, "Tr": 178, "kr_Vb": 2.33e-3, "tend": 48 * 60},
            7: {"M": 97, "ksi": 267, "kl": 1.77, "Tu": 91, "ku_Vi": 60e-3, "Tr": 50, "kr_Vb": 1.94e-3, "tend": 48 * 60},
            8: {"M": 73, "ksi": 200, "kl": 1.92, "Tu": 125, "ku_Vi": 60e-3, "Tr": 182, "kr_Vb": 2.38e-3, "tend": 48 * 60},
            9: {"M": 92, "ksi": 272, "kl": 1.71, "Tu": 87, "ku_Vi": 61e-3, "Tr": 49, "kr_Vb": 2.03e-3, "tend": 48 * 60},
            10: {"M": 74, "ksi": 191, "kl": 1.94, "Tu": 126, "ku_Vi": 61e-3, "Tr": 188, "kr_Vb": 2.47e-3, "tend": 48 * 60},
            11: {"M": 91, "ksi": 282, "kl": 1.67, "Tu": 85, "ku_Vi": 64e-3, "Tr": 48, "kr_Vb": 2.06e-3, "tend": 48 * 60},
        }

        # Ensure the patient ID is valid
        if pat not in patient_params:
            raise ValueError(f"Unknown patient ID: {pat}")

        # Set parameters for the given patient
        params = patient_params[pat]
        self.M = params["M"]
        self.ksi = params["ksi"]
        self.kl = params["kl"]
        self.Tu = params["Tu"]
        self.ku_Vi = params["ku_Vi"]
        self.Tr = params["Tr"]
        self.kr_Vb = params["kr_Vb"]
        self.tend = params["tend"]

        # Derived parameters
        self.Vb = 0.65 * self.M  # dL blood
        self.Vi = 2.5 * self.M  # dL for insulin
        self.kb = 128 / self.M  # Brain endogenous glucose consumption (mg glucose/dL/min)
        self.Ieq = (self.kl - self.kb) / self.ksi

    def params_to_inverse(self, pat):
        # Define trainable parameters for PINN optimization
        self.ksi = tf.Variable(self.ksi, dtype=tf.float32)
        return self.ksi

    def geometry_time(self):
        return dde.geometry.TimeDomain(self.min_t, self.max_t)
    
    def init_event_mappings(self, pat):
        # Define event timings (in minutes) and values for insulin and carbohydrate intake for all patients
        patient_insulin = {
            2: ([7.5 * 60, 12.5 * 60, 17 * 60, 24 * 60, 37 * 60, 37.5 * 60, 42.5 * 60, 48 * 60],
                [0.5, 2, 2, 22, 18, 17, 16, 19]),
            3: ([6 * 60, 18 * 60, 25 * 60],
                [1, 2, 3]),
            4: ([7 * 60, 13 * 60, 28 * 60, 35 * 60],
                [1.5, 2.5, 3, 1]),
            5: ([8 * 60, 22 * 60, 30 * 60, 45 * 60],
                [0.8, 2.2, 1.8, 2.5]),
            6: ([9 * 60, 12 * 60, 20 * 60, 37 * 60, 42 * 60],
                [1, 2, 2.5, 1.5, 1.2]),
            7: ([5 * 60, 10 * 60, 25 * 60, 35 * 60],
                [1.2, 1.8, 1.5, 2]),
            8: ([6 * 60, 18 * 60, 28 * 60, 40 * 60],
                [1, 2, 3, 2.5]),
            9: ([7 * 60, 12.5 * 60, 22 * 60, 30 * 60],
                [1.5, 2.2, 1.8, 2]),
            10: ([4 * 60, 14 * 60, 27 * 60, 38 * 60],
                [1, 2, 3, 1.5]),
            11: ([9 * 60, 18 * 60, 24 * 60, 32 * 60],
                [0.8, 2, 2.5, 2]),
        }

        patient_carbs = {
            2: ([24 * 60, 25.5 * 60, 37 * 60, 41 * 60, 42.5 * 60, 44.5 * 60],
                [128, 15, 150, 100, 7.5, 15]),
            3: ([20 * 60, 30 * 60],
                [100, 50]),
            4: ([12 * 60, 18 * 60, 34 * 60],
                [120, 60, 140]),
            5: ([14 * 60, 24 * 60, 36 * 60, 42 * 60],
                [110, 80, 130, 50]),
            6: ([15 * 60, 26 * 60, 39 * 60],
                [100, 120, 90]),
            7: ([10 * 60, 21 * 60, 33 * 60],
                [130, 75, 105]),
            8: ([18 * 60, 27 * 60, 40 * 60],
                [140, 100, 120]),
            9: ([12 * 60, 20 * 60, 35 * 60],
                [125, 80, 110]),
            10: ([14 * 60, 22 * 60, 30 * 60],
                [150, 90, 100]),
            11: ([16 * 60, 24 * 60, 36 * 60],
                [140, 70, 120]),
        }

        # Assign mappings for the given patient
        self.insulin_map = dict(zip(*patient_insulin.get(pat, ([], []))))
        self.carb_map = dict(zip(*patient_carbs.get(pat, ([], []))))

    def set_glucose_scaling(self, pat):
        # Scaling factor for glucose residuals based on initial glucose values
        X0_values = {
            2: [220, self.Ieq, 0, 0, 0],
            3: [125, self.Ieq, 0, 0, 0],
            4: [150, self.Ieq, 0, 0, 0],
            5: [140, self.Ieq, 0, 0, 0],
            6: [180, self.Ieq, 0, 0, 0],
            7: [160, self.Ieq, 0, 0, 0],
            8: [170, self.Ieq, 0, 0, 0],
            9: [155, self.Ieq, 0, 0, 0],
            10: [145, self.Ieq, 0, 0, 0],
            11: [130, self.Ieq, 0, 0, 0],
        }
        if pat not in X0_values:
            raise ValueError(f"Unknown patient ID: {pat}")

        # Set the glucose scaling factor dynamically
        self.glucose_scaling = X0_values[pat][0]

    def get_inputs(self, t):
        # Efficient lookup for insulin and carbohydrate values
        U = self.insulin_map.get(t, 1)  # Default to 1 if t not in insulin times
        R = self.carb_map.get(t, 0)    # Default to 0 if t not in carb times
        return U, R

    def ode(self, t, y):
        G = y[:, 0:1]
        I = y[:, 1:2]
        D = y[:, 2:]

        dG_dt = dde.grad.jacobian(G, t)
        d2I_dt2 = dde.grad.hessian(I, t)
        d2D_dt2 = dde.grad.hessian(D, t)

        # Get current U (insulin) and R (carbs) values
        U, R = self.get_inputs(t)

        eq_1 = (dG_dt - (-self.ksi * I) - D - self.kl + self.kb) / self.glucose_scaling
        eq_2 = d2I_dt2 - ((-1/self.Tu**2) * I) + ((-2/self.Tu) * dde.grad.jacobian(I, t)) + (U * (self.ku_Vi / self.Tu**2))
        eq_3 = d2D_dt2 - ((-1/self.Tr**2) * D) + ((-2/self.Tr) * dde.grad.jacobian(D, t)) + (R * (self.kr_Vb / self.Tr**2))

        return [eq_1, eq_2, eq_3]