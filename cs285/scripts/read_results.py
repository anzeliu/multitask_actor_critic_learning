from cProfile import label
import glob
import tensorflow as tf
import os
import time
import numpy as np
import matplotlib.pyplot as plt

def get_section_results_train_average_and_best(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    Best = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
            elif v.tag == 'Train_BestReturn':
                Best.append(v.simple_value)
    return X, Y, Best

def get_section_results_train(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

def get_section_results_eval(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

def print_return_per_step():
    logdir = 'data/q1_lb_rtg_na_CartPole-v0_27-09-2021_01-03-36/events*'
    eventfile = glob.glob(logdir)[0]

    X, Y = get_section_results_train(eventfile)
    for i, (x, y) in enumerate(zip(X, Y)):
        print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))

def get_average_results(logdir_list_dqn, logdir_list_doubledqn):
    """
        compute the average of section results from each file in the file_list
        save averaged results to numpy 
    """
    # dqn - compute the averaged performance from section results from files in file_list
    T_dqn = len(logdir_list_dqn)
    X_dqn = 0
    Y_average_dqn = 0
    for logdir in logdir_list_dqn:
        eventfile = glob.glob(logdir)[0]
        X_dqn, Y = get_section_results_train(eventfile)
        X_dqn, Y = np.array(X_dqn).astype(int), np.array(Y)
        Y_average_dqn = Y_average_dqn + Y
    Y_average_dqn = Y_average_dqn / T_dqn

    # doubledqn - compute the averaged performance from section results from files in file_list
    T_doubledqn = len(logdir_list_dqn)
    X_doubledqn = 0
    Y_average_doubledqn = 0
    for logdir in logdir_list_doubledqn:
        eventfile = glob.glob(logdir)[0]
        X_doubledqn, Y = get_section_results_train(eventfile)
        X_doubledqn, Y = np.array(X_doubledqn).astype(int), np.array(Y)
        Y_average_doubledqn = Y_average_doubledqn + Y
    Y_average_doubledqn = Y_average_doubledqn / T_doubledqn

    # set up file path to save the plot of average performance w.r.t time steps
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "q2_LunarLander-v3_averaged_performance_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'q2_LunarLander-v3_averaged_performance.png')

    # plot the average performance for both dqn and doubledqn in a single plot
    with open(file_name, 'wb') as f:
        plt.plot(X_dqn[:-1], Y_average_dqn, label="dqn average")
        plt.plot(X_doubledqn[:-1], Y_average_doubledqn, label="doubledqn average")
        plt.title("Averaged Performance of LunarLander-v3 with DQN and Double DQN")
        plt.xlabel("Train_EnvstepsSoFar")
        plt.ylabel("Train_AverageReturn")
        plt.legend()
        plt.savefig(file_name)

def sac_walker_plot():
    logdir = "data/sac_Walker2d_v4_Walker2d-v4_26-10-2022_01-26-36/events*"
    eventfile = glob.glob(logdir)[0]
    X, Y = get_section_results_eval(eventfile)
    plt.plot(X, Y)

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "sac_Walker2d-v4_performance_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'sac_Walker2d-v4_performance.png')

    plt.title("Performance of Walker2d-v4 with Soft Actor Critic")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.savefig(file_name)

def sac_halfcheetah_plot():
    logdir = "data/sac_HalfCheetah_v4_HalfCheetah-v4_25-10-2022_17-46-02/events*"
    eventfile = glob.glob(logdir)[0]
    X, Y = get_section_results_eval(eventfile)
    plt.plot(X, Y)

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "sac_HalfCheetah-v4_performance_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'sac_HalfCheetah-v4_performance.png')

    plt.title("Performance of HalfCheetah-v4 with Soft Actor Critic")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.savefig(file_name)

def sac_halfcheetah_after_walker_plot():
    logdir_list = ["data/sac_HalfCheetah_v4_HalfCheetah-v4_26-10-2022_01-26-54/events*", 
                    "data/sac_HalfCheetah_v4_HalfCheetah-v4_26-10-2022_02-48-28/events*"]
    labels = ["training alone", "with actor transfer"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "sac_HalfCheetah-v4_performance_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'sac_HalfCheetah-v4_performance.png')

    plt.title("Performance of HalfCheetah-v4 with Soft Actor Critic")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.savefig(file_name)

#############################################
# Multitask Learning
#############################################

# plot the eval average return of different variants of HalfCheetah using Advantage Actor Critic 
def actor_critic_halfcheetah_variants():
    logdir_list = ["data/HalfCheetah-v4_ntu1_ngstpu100_HalfCheetah-v4_06-12-2022_12-16-12/events*", 
                "data/HalfCheetah_A_ntu1_ngsptu100_HalfCheetah_A_06-12-2022_13-11-12/events*",
                "data/HalfCheetah_B_ntu1_ngsptu100_HalfCheetah_B_06-12-2022_13-48-08/events*",
                "data/HalfCheetah_C_ntu1_ngsptu100_HalfCheetah_C_06-12-2022_14-22-53/events*",
                "data/HalfCheetah_D_ntu1_ngsptu100_HalfCheetah_D_06-12-2022_14-24-46/events*"
                ]
    labels = ["Cheetah-v4", "Cheetah_A", "Cheetah_B", "Cheetah_C", "Cheetah_D"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_variants_performance_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_variants_performance_.png')

    plt.title("Performance of HalfCheetah Variants with Actor Critic")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.savefig(file_name)

# plot the eval average return of halfcheetah-v4 using single task model vs multitask model
def actor_critic_halfcheetah_v4_single_model():
    logdir_list = ["data/HalfCheetah-v4_ntu1_ngstpu100_HalfCheetah-v4_06-12-2022_12-16-12/events*"]
    labels = ["single task model"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_v4_single_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_v4_single_.png')

    plt.title("Performance of HalfCheetah-v4 with Single Task Model")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.savefig(file_name)

def actor_critic_halfcheetah_v4_multitask_model():
    logdir_list = ["data/Multitask_HalfCheetah-v4_1_100_08-12-2022_19-30-58_HalfCheetah-v4/events*"]
    labels = ["multitask model"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_v4_multitask_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_v4_multitask_.png')

    plt.title("Performance of HalfCheetah-v4 with Multitask Model")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.savefig(file_name)

def cheetah_v4_single_vs_multitask_model():
    logdir_list = ["data/HalfCheetah-v4_ntu1_ngstpu100_HalfCheetah-v4_06-12-2022_12-16-12/events*", 
                "data/Multitask_HalfCheetah_v4_A_08-12-2022_19-32-34_HalfCheetah-v4/events*"]
    labels = ["single task", "multitask learning with cheetah_A"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_v4_single_vs_multitask_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_v4_single_vs_multitask_.png')

    plt.title("Return of HalfCheetah-v4 trained alone vs trained with HalfCheetah_A")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.savefig(file_name)

def cheetah_A_single_vs_multitask_model():
    logdir_list = ["data/HalfCheetah_A_ntu1_ngsptu100_HalfCheetah_A_06-12-2022_13-11-12/events*", 
                "data/Multitask_HalfCheetah_v4_A_08-12-2022_19-32-34_HalfCheetah_A/events*"]
    labels = ["single task", "multitask learning with cheetah-v4"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_A_single_vs_multitask_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_A_single_vs_multitask_.png')

    plt.title("Return of HalfCheetah_A trained alone vs trained with HalfCheetah-v4")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.savefig(file_name)

def cheetah_v4_all_7_tasks():
    logdir_list = ["data/HalfCheetah-v4_ntu1_ngstpu100_HalfCheetah-v4_06-12-2022_12-16-12/events*", 
                "data/Multitask_all_7_tasks_08-12-2022_21-38-18_HalfCheetah-v4/events*"]
    labels = ["single task", "multitask"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_v4_single_vs_7_multitask_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_v4_single_vs_7_multitask_.png')

    plt.title("Return of HalfCheetah-v4 trained alone vs trained with all tasks")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(file_name)

def cheetah_A_all_7_tasks():
    logdir_list = ["data/HalfCheetah_A_ntu1_ngsptu100_HalfCheetah_A_06-12-2022_13-11-12/events*", 
                "data/Multitask_all_7_tasks_08-12-2022_21-38-18_HalfCheetah_A/events*"]
    labels = ["single task", "multitask"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_A_single_vs_7_multitask_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_A_single_vs_7_multitask_.png')

    plt.title("Return of HalfCheetah_A trained alone vs trained with all tasks")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(file_name)

def cheetah_B_all_7_tasks():
    logdir_list = ["data/HalfCheetah_B_ntu1_ngsptu100_HalfCheetah_B_06-12-2022_13-48-08/events*", 
                "data/Multitask_all_7_tasks_08-12-2022_21-38-18_HalfCheetah_B/events*"]
    labels = ["single task", "multitask"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_B_single_vs_7_multitask_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_B_single_vs_7_multitask_.png')

    plt.title("Return of HalfCheetah_B trained alone vs trained with all tasks")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(file_name)

def cheetah_C_all_7_tasks():
    logdir_list = ["data/HalfCheetah_C_ntu1_ngsptu100_HalfCheetah_C_06-12-2022_14-22-53/events*", 
                "data/Multitask_all_7_tasks_08-12-2022_21-38-18_HalfCheetah_C/events*"]
    labels = ["single task", "multitask"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_C_single_vs_7_multitask_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_C_single_vs_7_multitask_.png')

    plt.title("Return of HalfCheetah_C trained alone vs trained with all tasks")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(file_name)

def cheetah_D_all_7_tasks():
    logdir_list = ["data/HalfCheetah_D_ntu1_ngsptu100_HalfCheetah_D_06-12-2022_14-24-46/events*", 
                "data/Multitask_all_7_tasks_08-12-2022_21-38-18_HalfCheetah_D/events*"]
    labels = ["single task", "multitask"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_D_single_vs_7_multitask_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_D_single_vs_7_multitask_.png')

    plt.title("Return of HalfCheetah_D trained alone vs trained with all tasks")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(file_name)

def cheetah_E_all_7_tasks():
    logdir_list = ["data/HalfCheetah_E_ntu1_ngsptu100_HalfCheetah_E_09-12-2022_10-13-44/events*", 
                "data/Multitask_all_7_tasks_08-12-2022_21-38-18_HalfCheetah_E/events*"]
    labels = ["single task", "multitask"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_E_single_vs_7_multitask_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_E_single_vs_7_multitask_.png')

    plt.title("Return of HalfCheetah_E trained alone vs trained with all tasks")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(file_name)

def cheetah_F_all_7_tasks():
    logdir_list = ["data/HalfCheetah_F_ntu1_ngsptu100_HalfCheetah_F_09-12-2022_10-34-54/events*", 
                "data/Multitask_all_7_tasks_08-12-2022_21-38-18_HalfCheetah_F/events*"]
    labels = ["single task", "multitask"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_F_single_vs_7_multitask_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_F_single_vs_7_multitask_.png')

    plt.title("Return of HalfCheetah_F trained alone vs trained with all tasks")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(file_name)

# sharing actor
def cheetah_v4_architecture_2():
    logdir_list = ["data/HalfCheetah-v4_ntu1_ngstpu100_HalfCheetah-v4_06-12-2022_12-16-12/events*", 
                "data/Multitask_HalfCheetah_v4_A_12-12-2022_18-27-41_HalfCheetah-v4/events*"]
    labels = ["single task", "multitask"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_v4_multitask_architecture_2_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_v4_multitask_architecture_2.png')

    plt.title("Return of HalfCheetah-v4 trained with multitask architecture 2")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(file_name)

def cheetah_A_architecture_2():
    logdir_list = ["data/HalfCheetah_A_ntu1_ngsptu100_HalfCheetah_A_06-12-2022_13-11-12/events*", 
                "data/Multitask_HalfCheetah_v4_A_12-12-2022_18-27-41_HalfCheetah_A/events*"]
    labels = ["single task", "multitask"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_A_multitask_architecture_2_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_A_multitask_architecture_2.png')

    plt.title("Return of HalfCheetah_A trained with multitask architecture 2")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(file_name)

def cheetah_v4_architecture_3_beta():
    logdir_list = ["data/HalfCheetah-v4_ntu1_ngstpu100_HalfCheetah-v4_06-12-2022_12-16-12/events*", 
                "data/Multitask_Architecture_3_HalfCheetah_v4_A_beta_0.5_14-12-2022_08-58-39_HalfCheetah-v4/events*",
                "data/Multitask_Architecture_3_HalfCheetah_v4_A_beta_0.7_14-12-2022_09-18-43_HalfCheetah-v4/events*",
                "data/Multitask_Architecture_3_HalfCheetah_v4_A_beta_0.8_14-12-2022_13-05-54_HalfCheetah-v4/events*",
                "data/Multitask_Architecture_3_HalfCheetah_v4_A_beta_0.9_14-12-2022_13-06-02_HalfCheetah-v4/events*"]
    labels = ["single task", "multitask beta=0.5", "multitask beta=0.7", "multitask beta=0.8", "multitask beta=0.9"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_v4_multitask_architecture_3_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_v4_multitask_architecture_3.png')

    plt.title("Return of HalfCheetah-v4 trained with multitask architecture 3")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(file_name)

def cheetah_A_architecture_3_beta():
    logdir_list = ["data/HalfCheetah_A_ntu1_ngsptu100_HalfCheetah_A_06-12-2022_13-11-12/events*", 
                "data/Multitask_Architecture_3_HalfCheetah_v4_A_beta_0.5_A_14-12-2022_08-58-39_HalfCheetah_A/events*",
                "data/Multitask_Architecture_3_HalfCheetah_v4_A_beta_0.7_14-12-2022_09-18-43_HalfCheetah_A/events*",
                "data/Multitask_Architecture_3_HalfCheetah_v4_A_beta_0.8_14-12-2022_13-05-54_HalfCheetah_A/events*",
                "data/Multitask_Architecture_3_HalfCheetah_v4_A_beta_0.9_14-12-2022_13-06-02_HalfCheetah_A/events*"]
    labels = ["single task", "multitask beta=0.5", "multitask beta=0.7", "multitask beta=0.8", "multitask beta=0.9"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_A_multitask_architecture_3_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_A_multitask_architecture_3.png')

    plt.title("Return of HalfCheetah_A trained with multitask architecture 3")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(file_name)

def cheetah_v4_architecture_4_beta():
    logdir_list = ["data/HalfCheetah-v4_ntu1_ngstpu100_HalfCheetah-v4_06-12-2022_12-16-12/events*", 
                "data/Multitask_Architecture_4_HalfCheetah_v4_A_beta_0.5_14-12-2022_10-54-02_HalfCheetah-v4/events*",
                "data/Multitask_Architecture_4_HalfCheetah_v4_A_beta_0.7_14-12-2022_10-36-44_HalfCheetah-v4/events*"]
    labels = ["single task", "multitask beta=0.5", "multitask beta=0.7"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_v4_multitask_architecture_4_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_v4_multitask_architecture_4.png')

    plt.title("Return of HalfCheetah-v4 trained with multitask architecture 4")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(file_name)

def cheetah_A_architecture_4_beta():
    logdir_list = ["data/Multitask_Architecture_4_HalfCheetah_v4_A_beta_0.5_14-12-2022_10-54-02_HalfCheetah_A/events*", 
                "data/Multitask_Architecture_4_HalfCheetah_v4_A_beta_0.5_14-12-2022_10-54-02_HalfCheetah_A/events*",
                "data/Multitask_Architecture_4_HalfCheetah_v4_A_beta_0.7_14-12-2022_10-36-44_HalfCheetah_A/events*"]
    labels = ["single task", "multitask beta=0.5", "multitask beta=0.7"]
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results_eval(eventfile)
        plt.plot(X, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_A_multitask_architecture_4_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_A_multitask_architecture_4.png')

    plt.title("Return of HalfCheetah_A trained with multitask architecture 4")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(file_name)

# continual learning vs multitask learning
def continual_tasks_eval(n_iter, file):
    X = []
    Y = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                if n_iter == 0:
                    return X, Y
                else:
                    n_iter -= 1
                Y.append(v.simple_value)
    return X, Y

def continual_new_tasks_eval(n_iter, n_iter_tasks, file):
    X = []
    Y = [0 for _ in range(n_iter_tasks)]
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                if n_iter == n_iter_tasks:
                    return X, Y
                else:
                    n_iter -= 1
                Y.append(v.simple_value)
    return X, Y

def cheetah_v4_continual():
    logdir_list = ["data/HalfCheetah-v4_ntu1_ngstpu100_HalfCheetah-v4_06-12-2022_12-16-12/events*", 
                "data/Multitask_Architecture_3_HalfCheetah_v4_A_beta_0.8_14-12-2022_13-05-54_HalfCheetah-v4/events*",
                "data/Continual_Multitask_HalfCheetah_v4_A_beta_0.8_n_iter_tasks_20_14-12-2022_18-16-16_HalfCheetah-v4/events*"]
    labels = ["single task", "multitask", "continual multitask"]
    X_axis = []
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        if i <= 1:
            X, Y = get_section_results_eval(eventfile)
            X_axis = X
        else:
            X, Y = continual_tasks_eval(150, eventfile)
        plt.plot(X_axis, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_v4_multitask_vs_continual_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_v4_multitask_vs_continual_.png')

    plt.title("Return of HalfCheetah-v4 - continual - new task at 20 iter")    
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(file_name)

def cheetah_A_continual():
    logdir_list = ["data/HalfCheetah_A_ntu1_ngsptu100_HalfCheetah_A_06-12-2022_13-11-12/events*", 
                "data/Multitask_Architecture_3_HalfCheetah_v4_A_beta_0.8_14-12-2022_13-05-54_HalfCheetah_A/events*",
                "data/Continual_Multitask_HalfCheetah_v4_A_beta_0.8_n_iter_tasks_20_14-12-2022_18-16-16_HalfCheetah_A/events*"]
    labels = ["single task", "multitask", "continual multitask"]
    X_axis = []
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        if i <= 1:
            X, Y = get_section_results_eval(eventfile)
            X_axis = X
        else:
            X, Y = continual_new_tasks_eval(150, 20, eventfile)
        plt.plot(X_axis, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_A_multitask_vs_continual_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_A_multitask_vs_continual_.png')

    plt.title("Return of HalfCheetah_A - continual - new task at 20 iter")  
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(file_name)

def cheetah_v4_continual_encounter_newtask_at_50_iteration():
    logdir_list = ["data/HalfCheetah-v4_ntu1_ngstpu100_HalfCheetah-v4_06-12-2022_12-16-12/events*", 
                "data/Multitask_Architecture_3_HalfCheetah_v4_A_beta_0.8_14-12-2022_13-05-54_HalfCheetah-v4/events*",
                "data/Continual_Multitask_HalfCheetah_v4_A_beta_0.8_n_iter_tasks_50_14-12-2022_18-32-29_HalfCheetah-v4/events*"]
    labels = ["single task", "multitask", "continual multitask"]
    X_axis = []
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        if i <= 1:
            X, Y = get_section_results_eval(eventfile)
            X_axis = X
        else:
            X, Y = continual_tasks_eval(150, eventfile)
        plt.plot(X_axis, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_v4_multitask_vs_continual_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_v4_multitask_vs_continual_.png')

    plt.title("Return of HalfCheetah-v4 - continual - new task at 50 iter")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(file_name)

def cheetah_A_continual_encounter_newtask_at_50_iteration():
    logdir_list = ["data/HalfCheetah_A_ntu1_ngsptu100_HalfCheetah_A_06-12-2022_13-11-12/events*", 
                "data/Multitask_Architecture_3_HalfCheetah_v4_A_beta_0.8_14-12-2022_13-05-54_HalfCheetah_A/events*",
                "data/Continual_Multitask_HalfCheetah_v4_A_beta_0.8_n_iter_tasks_50_14-12-2022_18-32-29_HalfCheetah_A/events*"]
    labels = ["single task", "multitask", "continual multitask"]
    X_axis = []
    for i in range(len(logdir_list)):
        logdir = logdir_list[i]
        eventfile = glob.glob(logdir)[0]
        if i <= 1:
            X, Y = get_section_results_eval(eventfile)
            X_axis = X
        else:
            X, Y = continual_new_tasks_eval(150, 50, eventfile)
        plt.plot(X_axis, Y, label=labels[i])
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graph')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = "actor_critic_cheetah_A_multitask_vs_continual_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    file_name = os.path.join(logdir, 'actor_critic_cheetah_A_multitask_vs_continual_.png')

    plt.title("Return of HalfCheetah_A - continual - new task at 50 iter")
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(file_name)

if __name__ == '__main__':
    import glob
    plot_functions = [
                    actor_critic_halfcheetah_variants,
                    actor_critic_halfcheetah_v4_single_model,
                    actor_critic_halfcheetah_v4_multitask_model,
                    cheetah_v4_single_vs_multitask_model,
                    cheetah_A_single_vs_multitask_model,
                    cheetah_v4_all_7_tasks,
                    cheetah_A_all_7_tasks,
                    cheetah_B_all_7_tasks,
                    cheetah_C_all_7_tasks,
                    cheetah_D_all_7_tasks,
                    cheetah_E_all_7_tasks,
                    cheetah_F_all_7_tasks,
                    cheetah_v4_architecture_2,
                    cheetah_A_architecture_2,
                    cheetah_v4_architecture_3_beta,
                    cheetah_A_architecture_3_beta,
                    cheetah_v4_architecture_4_beta,
                    cheetah_A_architecture_4_beta,
                    cheetah_v4_continual,
                    cheetah_A_continual,
                    cheetah_v4_continual_encounter_newtask_at_50_iteration,
                    cheetah_A_continual_encounter_newtask_at_50_iteration
                    ]
    for i in range(len(plot_functions)):
        plt.figure(i)
        plot_functions[i]()
    print("all plots complete! \n")




    


    

    