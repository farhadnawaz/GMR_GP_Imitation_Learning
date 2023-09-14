#! /usr/bin/env python
##################################################
# Author: Noemie Jaquier, 2019
# License: MIT
# Contact: noemie.jaquier@idiap.ch
##################################################

import numpy as np
import GPy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import os.path
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat # loading data from matlab
from utils.gmr import Gmr
from utils.gmr import plot_gmm
from utils.gp_coregionalize_with_mean_regression import GPCoregionalizedWithMeanRegression
from utils.gmr_mean_mapping import GmrMeanMapping
from utils.gmr_kernels import Gmr_based_kernel
import argparse
import time
from fastdtw import fastdtw
import warnings   
warnings.filterwarnings("ignore")

# GMR-based GPR on 3D trajectories with time as input
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='Spiral', type=str)
    parser.add_argument('--n_gaussian', default=10, type=int)
    parser.add_argument('--n_test_execution', default=5, type=int)
    args = parser.parse_args()
    data_name = args.data_name

    demos = np.load('3D_robot_periodic/'+args.data_name+'.npy')[:,:,:3]
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.set_aspect('equal')
    # ax.plot(demos[0][:,0], demos[0][:,1], demos[0][:,2], color='blue')
    # plt.show()

    # Parameters
    nb_data = demos[0].shape[0]
    nb_data_sup = 0
    nb_samples = 1
    dt = 0.0033
    input_dim = 1
    output_dim = 3
    in_idx = [0]
    out_idx = [1, 2, 3]
    nb_states = args.n_gaussian

    nb_prior_samples = args.n_test_execution
    nb_posterior_samples = 1

    # Create time data
    demos_t = [np.arange(demos[i].shape[0])[:, None] + 1 for i in range(nb_samples)]
    # Stack time and position data
    demos_tx = [np.hstack([demos_t[i]*dt, demos[i]]) for i in range(nb_samples)]

    # Stack demos
    demos_np = demos_tx[0]
    for i in range(1, nb_samples):
        demos_np = np.vstack([demos_np, demos_tx[i]])

    X = demos_np[:, 0][:, None]
    Y = demos_np[:, 1:]

    # Train data for GPR
    X_list = [np.hstack((X, X)) for i in range(output_dim)]
    Y_list = [Y[:, i][:, None] for i in range(output_dim)]

    # Test data
    Xt = dt * np.arange(demos[0].shape[0] + nb_data_sup)[:, None]
    nb_data_test = Xt.shape[0]
    Xtest, _, output_index = GPy.util.multioutput.build_XY([np.hstack((Xt, Xt)) for i in range(output_dim)])


    begin_train = time.time()
    # GMM
    gmr_model = Gmr(nb_states=nb_states, nb_dim=input_dim + output_dim, in_idx=in_idx, out_idx=out_idx)
    gmr_model.init_params_kbins(demos_np.T, nb_samples=nb_samples)
    gmr_model.gmm_em(demos_np.T)

    # GMR prediction
    mu_gmr = []
    sigma_gmr = []
    for i in range(Xt.shape[0]):
        mu_gmr_tmp, sigma_gmr_tmp, H_tmp = gmr_model.gmr_predict(Xt[i])
        mu_gmr.append(mu_gmr_tmp)
        sigma_gmr.append(sigma_gmr_tmp)

    mu_gmr = np.array(mu_gmr)
    sigma_gmr = np.array(sigma_gmr)

    # Define GPR likelihood and kernels
    likelihoods_list = [GPy.likelihoods.Gaussian(name="Gaussian_noise_%s" %j, variance=0.01) for j in range(output_dim)]
    # kernel_list = [GPy.kern.RBF(1, variance=1., lengthscale=0.1) for i in range(gmr_model.nb_states)]
    kernel_list = [GPy.kern.Matern52(1, variance=1., lengthscale=5.) for i in range(gmr_model.nb_states)]

    # Fix variance of kernels
    for kernel in kernel_list:
        kernel.variance.fix(1.)
        kernel.lengthscale.constrain_bounded(0.01, 10.)

    # Bound noise parameters
    for likelihood in likelihoods_list:
        likelihood.variance.constrain_bounded(0.001, 0.05)

    # GPR model
    K = Gmr_based_kernel(gmr_model=gmr_model, kernel_list=kernel_list)
    mf = GmrMeanMapping(2*input_dim+1, 1, gmr_model)

    m = GPCoregionalizedWithMeanRegression(X_list, Y_list, kernel=K, likelihoods_list=likelihoods_list, mean_function=mf)

    # Parameters optimization
    m.optimize('bfgs', max_iters=100, messages=True)

    # Print model parameters
    #print(m)
    print(" ")
    print("******* Data "+ data_name +" *******")
    print(" ")
    print("#######################################")
    print("Training time",time.time() - begin_train)
    print("#######################################")
    # GPR prior (no observations)

    begin_prior = time.time()
    prior_traj = []
    prior_mean = mf.f(Xtest)[:, 0]
    prior_kernel = m.kern.K(Xtest)
    prior_time = time.time() - begin_prior
    total_sampling_time = 0.0
    for i in range(nb_prior_samples):
        one_sample_begin = time.time()
        prior_traj_tmp = np.random.multivariate_normal(prior_mean, prior_kernel)
        prior_traj.append(np.reshape(prior_traj_tmp, (output_dim, -1)))
        total_sampling_time += time.time() - one_sample_begin
    print("#######################################")
    print("Testing time",total_sampling_time/nb_prior_samples + prior_time)
    print("#######################################")
    prior_kernel_tmp = np.zeros((nb_data_test, nb_data_test, output_dim * output_dim))
    for i in range(output_dim):
        for j in range(output_dim):
            prior_kernel_tmp[:, :, i * output_dim + j] = prior_kernel[i * nb_data_test:(i + 1) * nb_data_test, j * nb_data_test:(j + 1) * nb_data_test]
    prior_kernel_rshp = np.zeros((nb_data_test, output_dim, output_dim))
    for i in range(nb_data_test):
        prior_kernel_rshp[i] = np.reshape(prior_kernel_tmp[i, i, :], (output_dim, output_dim))




    # Priors
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.set_aspect('equal')
    # plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3.)
    # plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color=[0.20, 0.54, 0.93], marker='X', s=80)
    # plot_gmm(mu_gmr, prior_kernel_rshp, alpha=0.05, color=[0.64, 0.27, 0.73])
    ax.plot(demos[0][:,0], demos[0][:,1], demos[0][:,2], color='blue', label='Demonstration')
    total_dtw = 0.0
    for i in range(nb_prior_samples):
        ax.plot(prior_traj[i][0], prior_traj[i][1], prior_traj[i][2], linewidth=1., label='Reproduction')
        ax.scatter3D(prior_traj[i][0, 0], prior_traj[i][1, 0], prior_traj[i][2,0], marker='X', s=80)
        # np.save('results/'+letter+'/'+letter+str(args.n_gaussian)+'.npy', prior_traj[i])
        # np.save('results/'+letter+'/demo.npy', demos[0])
        distance, path = fastdtw(demos[0, :, :3], prior_traj[i].T)
        total_dtw += distance
    print("#######################################")
    print("DTW",total_dtw / nb_prior_samples)
    print("#######################################")
    # plt.xlabel('$y_1$', fontsize=30)
    # plt.ylabel('$y_2$', fontsize=30)
    # plt.locator_params(nbins=3)
    # plt.tick_params(labelsize=20)
    # plt.tight_layout()
    # plt.savefig('results/'+letter+'/GMRbGP_gmm_'+str(args.n_gaussian)+'priors_datasup.png')
    ax.legend()
    plt.show()