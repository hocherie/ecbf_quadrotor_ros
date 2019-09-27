#!/usr/bin/env python
import numpy as np
import rospy
import time
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from ca_control_msgs.msg import RPYrVertv
from nav_msgs.msg import Odometry
from cvxopt import matrix
from cvxopt import solvers

ecbf_a = 1
ecbf_b = 1
safety_dist = 1
class EcbfNode:
    """
    base class for processing ecbf control given 
    """
    def __init__(self):
        # Declare class members
        self.att_pub = rospy.Publisher('/drone_cmd', RPYrVertv, queue_size=10)
        
        # Declare subscribers
        rospy.Subscriber('/dji_sdk/odometry', Odometry, self.odom_cb)

        mock_state = {"x": np.array([5, 0, 10]),
                "xdot": np.zeros(3,) }
        self.ecbf = ECBF_control(mock_state)


    def odom_cb(self, odom_msg):
        start_time = time.time()
    #     img_batch_1 = self.process_image(img_msg)
    #     net_vel = self.vel_regressor.predict_velocities(img_batch_1)
    #     net_vel[0:2] = net_vel[0:2]*1.0
	# net_vel[3] = net_vel[3]*1.0
    #     vel_cmd = TwistStamped()
    #     vel_cmd.twist.linear.x = net_vel[0]
    #     vel_cmd.twist.linear.y = net_vel[1]
    #     vel_cmd.twist.linear.z = net_vel[2]
    #     vel_cmd.twist.angular.z = net_vel[3]
    #     elapsed_time_net = time.time() - start_time
    #     #print('Avg network elapsed time: {} ms'.format(elapsed_time_net))
        # self.vel_pub.publish(vel_cmd)
        new_obs = np.array([[0.5], [5]])
        # print(odom_msg)
        ## ecbf

        # update ecbf state
        pos = odom_msg.pose.pose.position
        vel = odom_msg.twist.twist.linear
        # ecbf.state = state 
        self.ecbf.state["x"] = np.array([pos.x, pos.y, pos.z])
        self.ecbf.state["xdot"] = np.array([vel.x, vel.y, vel.z])
        u_hat_acc = self.ecbf.compute_safe_control(obs=new_obs)
        u_hat_acc = np.vstack((u_hat_acc,[0]))
        print("desired accel")
        print(u_hat_acc)
        des_theta = self.dynamic_inversion(u_hat_acc, self.ecbf.state)
        # print(np.array((des_theta)))
        # print(np.degrees(np.array((des_theta))))
        des_deg = np.degrees(np.array((des_theta)))
        elapsed_time_net = time.time() - start_time
        
        print('ECBF elapsed time: {} ms'.format(elapsed_time_net))

        # vel_
        drone_cmd = RPYrVertv()

        drone_cmd.roll = -des_deg[0]
        drone_cmd.pitch = -des_deg[1]
        print(drone_cmd.roll, drone_cmd.pitch)
        self.att_pub.publish(drone_cmd)

    def dynamic_inversion(self, des_acc, state):
        """Invert dynamics. For outer loop, given v_tot, compute attitude.
        Similar to control allocator.
        TODO: do 1-1 mapping?
        Parameters
        ----------
        self.v_tot
            total v: v_cr + v_lc - v_ad
        state #TODO: use self.x
            state
        Returns
        -------
        desired_theta: np.ndarray(3,)
            desired roll, pitch, yaw angle (rad) to attitude controller
        """
        yaw = 0 # TODO
        g = -9.81 # TODO
        U1 = np.linalg.norm(des_acc - np.array([0, 0, g]))
        des_pitch_noyaw = np.arcsin(des_acc[0] / U1)
        des_angle = [des_pitch_noyaw,
                    np.arcsin(des_acc[1] / (U1 * np.cos(des_pitch_noyaw)))]
        des_pitch = des_angle[0] * np.cos(yaw) + des_angle[1] * np.sin(yaw)
        des_roll = des_angle[0] * np.sin(yaw) - des_angle[1] * np.cos(yaw)
        # print(des_pitch[0], des_roll[0], "hi")

        # TODO: move to attitude controller?
        des_pitch = np.clip(des_pitch, np.radians(-30), np.radians(30))
        des_roll = np.clip(des_roll, np.radians(-30), np.radians(30))

        # TODO: currently, set yaw as constant
        des_yaw = yaw
        des_theta = [des_roll[0], des_pitch[0], des_yaw]
        # assert()

        return des_theta 

class ECBF_control():
    def __init__(self, state, goal=np.array([[0], [10]])):
        self.state = state
        self.shape = {"a":1, "b":1, "safety_dist":safety_dist} #TODO: a, b
        Kp = 3
        Kd = 4
        self.K = np.array([Kp, Kd])
        self.goal=goal
        self.use_safe = True
        # pass

    def compute_h(self, obs=np.array([[0], [0]]).T):
        rel_r = np.atleast_2d(self.state["x"][:2]).T - obs
        # TODO: a, safety_dist, obs, b
        hr = self.h_func(rel_r[0], rel_r[1], self.shape['a'], self.shape["b"], safety_dist)
        return hr

    def compute_hd(self, obs):
        rel_r = np.atleast_2d(self.state["x"][:2]).T - obs
        rd = np.atleast_2d(self.state["xdot"][:2]).T
        term1 = (4 * np.power(rel_r[0],3) * rd[0])/(np.power(self.shape['a'],4))
        term2 = (4 * np.power(rel_r[1],3) * rd[1])/(np.power(self.shape['b'],4))
        return term1+term2

    def compute_A(self, obs):
        rel_r = np.atleast_2d(self.state["x"][:2]).T - obs
        A0 = (4 * np.power(rel_r[0], 3))/(np.power(self.shape['a'], 4))
        A1 = (4 * np.power(rel_r[1], 3))/(np.power(self.shape['b'], 4))

        return np.array([np.hstack((A0, A1))])

    def compute_h_hd(self, obs):
        h = self.compute_h(obs)
        hd = self.compute_hd(obs)

        return np.vstack((h, hd)).astype(np.double)

    def compute_b(self, obs):
        """extra + K * [h hd]"""
        rel_r = np.atleast_2d(self.state["x"][:2]).T - obs
        rd = np.array(np.array(self.state["xdot"])[:2])
        extra = -(
            (12 * np.square(rel_r[0]) * np.square(rd[0]))/np.power(self.shape['a'],4) +
            (12 * np.square(rel_r[1]) * np.square(rd[1]))/np.power(self.shape['b'], 4)
        )

        b_ineq = extra - np.dot(self.K, self.compute_h_hd(obs))
        return b_ineq

    def compute_safe_control(self,obs):
        if self.use_safe:
            A = self.compute_A(obs)
            assert(A.shape == (1,2))

            b_ineq = self.compute_b(obs)

            #Make CVXOPT quadratic programming problem
            P = matrix(np.eye(2), tc='d')
            q = -1 * matrix(self.compute_nom_control(), tc='d')
            G = -1 * matrix(A.astype(np.double), tc='d')

            h = -1 * matrix(b_ineq.astype(np.double), tc='d')
            solvers.options['show_progress'] = False
            sol = solvers.qp(P,q,G, h, verbose=False) # get dictionary for solution

            optimized_u = sol['x']

        else:
            optimized_u = self.compute_nom_control()


        return optimized_u

    def compute_nom_control(self, Kn=np.array([-0.1, -0.2])):
        #! mock
        vd = Kn[0]*(np.atleast_2d(self.state["x"][:2]).T - self.goal)
        # print("error", (np.atleast_2d(self.state["x"][:2]).T - self.goal))
        # u_nom = vd
        u_nom = Kn[1]*(np.atleast_2d(self.state["xdot"][:2]).T - vd)
        print("error_v", vd)
        if np.linalg.norm(u_nom) > 1:
            u_nom = (u_nom/np.linalg.norm(u_nom))
        return u_nom.astype(np.double)

    
    def h_func(self, r1, r2, a, b, safety_dist):
        hr = np.power(r1,4)/np.power(a, 4) + \
            np.power(r2, 4)/np.power(b, 4) - safety_dist
        return hr

# def main():

#     new_obs = np.array([[0], [1]])
#     ecbf.state = state # update ecbf state
#     u_hat_acc = ecbf.compute_safe_control(obs=new_obs)
    # # u_hat_acc = np.ndarray.flatten(np.array(np.vstack((u_hat_acc,np.zeros((1,1))))))  # acceleration
    #     assert(u_hat_acc.shape == (3,))
    #     u_motor = go_to_acceleration(state, u_hat_acc, dyn.param_dict) # desired motor rate ^2

    #     state = dyn.step_dynamics(state, u_motor)
    #     ecbf.state = state
    #     state_hist.append(state["x"])
    #     if(tt % 100 == 0):
    #         print(tt)
    #         plt.cla()
    #         state_hist_plot = np.array(state_hist)
    #         nom_cont = ecbf.compute_nom_control()
    #         plt.plot([state_hist_plot[-1, 0], state_hist_plot[-1, 0] + 100 *
    #                   u_hat_acc[0]],
    #                  [state_hist_plot[-1, 1], state_hist_plot[-1, 1] + 100 * u_hat_acc[1]], label="Safe")
    #         plt.plot([state_hist_plot[-1, 0], state_hist_plot[-1, 0] + 100 *
    #                   nom_cont[0]],
    #                  [state_hist_plot[-1, 1], state_hist_plot[-1, 1] + 100 * nom_cont[1]],label="Nominal")

    #         plt.plot(state_hist_plot[:, 0], state_hist_plot[:, 1],'k')
    #         plt.plot(ecbf.goal[0], ecbf.goal[1], '*r')
    #         plt.plot(state_hist_plot[-1, 0], state_hist_plot[-1, 1], '*k') # current

    #         ecbf.plot_h(new_obs)
if __name__ == '__main__':
    rospy.init_node('ecbf_node', log_level=rospy.DEBUG)
    rospy.loginfo('ecbf_node initialized')
    node = EcbfNode()
    rospy.spin()
