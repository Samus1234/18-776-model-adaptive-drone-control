# 18-776-model-adaptive-drone-control

**A Python implementation of a 6-DoF quad-rotor drone simulation and 3 nonlinear model adaptive controllers with Extended and Unscented Kalman Filters for state estimation**

**Please refer to the "Nonlinear_Final_Report.pdf" for more details on the drone modeling and an extensive performance analysis of each controller.**

## External Dependecies
* Python with Numpy, Matplotlib, Scipy, and control packages

## Structure
* **The project is composed of three main parts. First, we model the dynamics of the quadcopter. Second, we analyze and develop nonlinear controllers to track a figure-8 reference trajectory parameterized by time. Third, we study their characteristics and juxtapose their performance with the previously developed linear control algorithms. More specifically, we analyze the mean square tracking (and estimation error where necessary) and examine the ultimate bound of each controller for a given exogenous disturbance such as wind. Every Python file includes the full drone model derived using the Euler-Lagrange formulation and EKF and UKF state estimators**

* droneMPC.py - Contains a lifted linear model predictive controller that applies constraints on the maximum and minimum rotor RPMs.
* droneMPCFeasibility.py - Studies the impact of state contraints on the feasibility of the model predictive controller.
* droneDynamicInversionControl.py - Implements a nonlinear dynamic inversion controller and an inner PID controller to control the linearized and inverted drone model.
* dynamicInversionWithParameterIdentification - Implements a nonlinear dynamic inversion controller with online parameter identification of exogenous wind disturbance for added robustness.

## Demonstration

**Model Predictive Controller with control contraints only**

![](https://github.com/sidd-1234/18-776-model-adaptive-drone-control/blob/main/droneMPCWithControlConstraintsOnly.gif)

**Model Predictive Controller with state and control constraints**

![](https://github.com/sidd-1234/18-776-model-adaptive-drone-control/blob/main/droneMPCWithStateConstraints.gif)

**Nonlinear Dynamic Inversion Controller**

![](https://github.com/sidd-1234/18-776-model-adaptive-drone-control/blob/main/droneNDI.gif)

**Nonlinear Dynamic Inversion Controller with model adaptation using online parameter estimation**

![](https://github.com/sidd-1234/18-776-model-adaptive-drone-control/blob/main/droneAdaptiveNDI.gif)
