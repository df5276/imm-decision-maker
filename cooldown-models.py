from filterpy.kalman import KalmanFilter, IMMEstimator
from scipy.signal import cont2discrete
import pdb
import numpy as np
import control as ct
import matplotlib.pyplot as plt

def first_order_sys(K, tau, inputs=None, outputs=None):
    return ct.tf(K, [tau, 1], inputs=inputs, outputs=outputs)

dt = 0.1
t_vec = np.arange(0,10,dt)

def temperature_history():
    u = np.zeros(t_vec.shape)
    # window opens in middle for a while
    u[(4 < t_vec) & (t_vec < 7)] = -1

    sys_c = first_order_sys(1,0.1)
    yout = ct.forced_response(sys_c, t_vec, inputs=u)
    # Crudely add the initial level back to simulated output
    temp = yout.outputs + 5.0
#    fig, ax, = plt.subplots()
#    ax.plot(temp)
    return temp



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
        self.R = model.R
        self.Q = .1
        self.reset() # Set the covariance matrix.

    def reset(self):
        self.P = np.identity(self.dim_x)*1000


    def run(self, meas):
        est_state = np.zeros(t_vec.shape+(self.dim_x,1)) # x has dim_x rows and 1 column
        cov = np.zeros(t_vec.shape+(self.dim_x, self.dim_x)) # cov is x by x
        log_likelihood = np.zeros(t_vec.shape)
       
        for i, z in enumerate(meas):
            self.predict()
            self.update(z)
            log_likelihood[i] = self.log_likelihood
            est_state[i] = self.x
            cov[i] =  self.P

        self.reset() # Put the covariance back to the initial state, to avoid the 'run' function
        # having side effects.

        return est_state, cov, log_likelihood

def main():
   
    cooling_model = Model(1)
    estimator = Estimator(cooling_model)
    
    stationary_model = Model(1000)

    plot(estimator, cooling_model, "cooling_model")
    plot(estimator, stationary_model, "stationary_model")


    # IMM setup 
    R = 0.01
    temp_hist = temperature_history()
    temp_data = temp_hist + np.random.normal(0, np.sqrt(R), len(t_vec))
        
    filters = [Estimator(cooling_model), Estimator(stationary_model)]
    
    M = np.array([[0.5, 0.5],
                  [0.5, 0.5]]) # This is the matrix that describes model transition likelihood

    mu = np.array([5.0,5.0]) # Initital state vector
    bank = IMMEstimator(filters, mu, M)

    probs = np.zeros(t_vec.shape+(1,2))
    for i, z in enumerate(temp_data):
        bank.predict()
        bank.update(z)
        probs[i] = bank.mu.copy()

    fig, ax = plt.subplots(2,1)
    ax[0].plot(t_vec, temp_data)
    ax[1].plot(t_vec, probs[:,0,0], label="cooling")
    ax[1].plot(t_vec, probs[:,0,1], label="stationary")
    ax[0].legend()




    plt.show()


def plot(estimator, model, name=None):
    est_state, cov, log_likelihood  = estimator.run(model.meas)
    est_state = est_state[:,0,0]
    cov = cov[:,0,0]
    fig, ax  = plt.subplots(2,1)
    ax[0].set_title("model: {}, log likelihood = {}".format(name, np.sum(log_likelihood)))
    ax[0].plot(t_vec,model.meas)
    ax[0].plot(t_vec, est_state)
    ax[0].fill_between(t_vec, est_state - cov, est_state + cov, alpha=0.2)
    ax[1].plot(t_vec, log_likelihood)



if __name__ == "__main__":
    main()
