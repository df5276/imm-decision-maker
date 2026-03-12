from filterpy.kalman import KalmanFilter
from scipy.signal import cont2discrete
import pdb
import numpy as np
import control as ct
import matplotlib.pyplot as plt

def first_order_sys(K, tau, inputs=None, outputs=None):
    return ct.tf(K, [tau, 1], inputs=inputs, outputs=outputs)

dt = 0.1
t_vec = np.arange(0,10,dt)

class Model:
    def __init__(self, tau, x0=5.0, R=1.0):
        """Model linear system
        tau - first order time constant
        x0 - initial state
        R - covariance of the measurement"""

        self.tau = tau
        self.x0 = x0
        self.R = R

        # Create discrete linear system
        sys_c = ct.tf2ss(first_order_sys(1,tau))
        sys_d = ct.c2d(sys_c, dt)

        # There's a bug in scipy, the C matrix becomes weird
        sys_d.C=[[1]]
        self.sys_d = sys_d
        self.yout, self.meas = self.sim()

    def sim(self):
        """Simulate measurements of the linear system

        Returns:
        yout - ground truth sim
        meas - simulated measurements"""

        yout = ct.forced_response(self.sys_d, t_vec, initial_state=self.x0)

        # Sample Gaussian for measurement
        samples  = np.random.normal(0, np.sqrt(self.R), len(t_vec))
        meas = yout.outputs+samples

        return yout, meas

class Estimator(KalmanFilter):
    def __init__(self, model):
        self.model = model

        # Define Kalman filter
        super().__init__(dim_x=1, dim_z=1)
        self.F = model.sys_d.A
        self.x = np.array([[model.x0]])
        self.H = np.array([[1]])
        self.P *= 1000
        self.R = model.R
        self.Q = .1

    def run(self):
        est_state = np.zeros(t_vec.shape)
        cov = np.zeros(t_vec.shape)
       
        for i, z in enumerate(model.meas):
            self.predict()
            self.update(z)
            est_state[i] = self.x
            cov[i] =  self.P

        return est_state, cov


model = Model(1)
estimator = Estimator(model)
est_state, cov = estimator.run()

fig, ax  = plt.subplots(2,1)
ax[0].set_title("log likelihood = {}".format(estimator.log_likelihood))
ax[0].plot(t_vec,model.meas)
ax[0].plot(t_vec, est_state)
ax[0].fill_between(t_vec, est_state - cov, est_state + cov, alpha=0.2)


ax[1].plot(t_vec, cov)


plt.show()
