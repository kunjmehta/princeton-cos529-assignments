import numpy as np
import pickle
import argparse

# new imports
import sys
import copy
np.set_printoptions(threshold=sys.maxsize, suppress=True)

# resourcing - UNIX
# import resource
# import signal

# Uses equation 1.7 in the notes part 3 (LM equation without rescaling)
def LM(solution, error, n_cameras, n_points, n_observations):

    if solution['is_calibrated']:
        num_variables = 6
    else:
        num_variables = 7

    JTe = np.zeros(num_variables*n_cameras + 3*n_points)
    JTJ = np.zeros((num_variables*n_cameras + 3*n_points,  num_variables*n_cameras + 3*n_points))

    for index, camera_point in enumerate(solution['observations']):
        camera_id = camera_point[0]
        point_id = camera_point[1]
        f = solution['focal_lengths'][camera_id]
        J_r1 = np.zeros(num_variables*n_cameras + 3*n_points)
        J_r2 = np.zeros(num_variables*n_cameras + 3*n_points)
        cam_loc = num_variables * camera_id
        pt_loc = num_variables * n_cameras + 3 * point_id

        P = solution['points'][point_id]
        R = solution['poses'][camera_id]

        _, point_prime = projection(solution, camera_point[0], camera_point[1])

        # Two equations:
        # 1. Projected point P' = R(exp(psi^))P + t where R is rotation matrix, ^ is cross-product, t is translation matrix and P = [X Y Z] 3D world co-ordinates
        # P' = [X' Y' Z']
        # 2. actual image pixel co-ordinates p = (u, v) = [f * X'/Z', f * Y'/Z']

        # derivative of projected point with respect to translation vector dP' / dt
        dPprimedt = 1
        # derivative of projected point with respect to world co-ordinates P dP' / dP at psi=0
        R = R[0:3, 0:3]
        dPprimedP = R

        # derivative of projected point with respect to psi dP'/dpsi = (-RP)^ where ^ is cross product
        # Notes 2 - Equation 1.17
        # Phat = cross_product(P)
        # dPprimedpsi = np.dot(-R, Phat)
        dPprimedpsi = cross_product(np.dot(-R, P))

        # derivative of image pixels with respect to projected point du/dP' and dv/dP'
        # du/dP' = [du/dX' du/dY' du/dZ']
        # dv/dP' = [dv/dX' dv/dY' dv/dZ']
        dudPprime = np.array([f/point_prime[2], 0, -f*point_prime[0] / (point_prime[2]*point_prime[2])])
        dvdPprime = np.array([0, f/point_prime[2], -f*point_prime[1] / (point_prime[2]*point_prime[2])])

        # derivative of image pixels with respect to psi du/dpsi = du/dP' * dP'/dsi
        dudpsi = np.dot(dudPprime, dPprimedpsi)
        dvdpsi = np.dot(dvdPprime, dPprimedpsi)

        # derivative of image pixels with respect to translation matrix du/dt = du/dP' * dP'/dt
        dudt = dudPprime * dPprimedt
        dvdt = dvdPprime * dPprimedt

        # derivative of image pixels with respect to rotation matrix du/dR = du/dP' * dP'/dR
        dudP = np.dot(dudPprime, R)
        dvdP = np.dot(dvdPprime, R)

        # derivative of image pixels with respect to focal length if not calibrated 
        if not solution['is_calibrated']:
            dudf = point_prime[0]/point_prime[2]
            dvdf = point_prime[1]/point_prime[2]

        J_r1[cam_loc:cam_loc+3] = dudpsi
        J_r1[cam_loc+3:cam_loc+6] = dudt
        if not solution['is_calibrated']:
            J_r1[cam_loc+6] = dudf
            
        J_r2[cam_loc:cam_loc+3] = dvdpsi
        J_r2[cam_loc+3:cam_loc+6] = dvdt
        if not solution['is_calibrated']:
            J_r2[cam_loc+6] = dvdf

        J_r1[pt_loc:pt_loc+3] = dudP
        J_r2[pt_loc:pt_loc+3] = dvdP

        JTe += error[2*index] * J_r1
        JTe += error[2*index+1] * J_r2 
        
        if not solution['is_calibrated']:
            cam_derivu = np.reshape(np.hstack([dudpsi, dudt, dudf]), (-1, 1))
        else:
            cam_derivu = np.reshape(np.hstack([dudpsi, dudt]), (-1, 1))

        if not solution['is_calibrated']:
            cam_derivv = np.reshape(np.hstack([dvdpsi, dvdt, dvdf]), (-1, 1))
        else:
            cam_derivv = np.reshape(np.hstack([dvdpsi, dvdt]), (-1, 1))

        pt_derivv = np.reshape(dvdP, (-1, 1))
        JTJ[cam_loc:cam_loc+num_variables, cam_loc:cam_loc+num_variables] += np.dot(cam_derivv, cam_derivv.T)
        JTJ[cam_loc:cam_loc+num_variables, pt_loc:pt_loc+3] += np.dot(cam_derivv, pt_derivv.T)
        JTJ[pt_loc:pt_loc+3, cam_loc:cam_loc+num_variables] += np.dot(pt_derivv, cam_derivv.T)
        JTJ[pt_loc:pt_loc+3, pt_loc:pt_loc+3] += np.dot(pt_derivv, pt_derivv.T)

        pt_derivu = np.reshape(dudP, (-1, 1))
        JTJ[cam_loc:cam_loc+num_variables, cam_loc:cam_loc+num_variables] += np.dot(cam_derivu, cam_derivu.T)
        JTJ[cam_loc:cam_loc+num_variables, pt_loc:pt_loc+3] += np.dot(cam_derivu, pt_derivu.T)
        JTJ[pt_loc:pt_loc+3, cam_loc:cam_loc+num_variables] += np.dot(pt_derivu, cam_derivu.T)
        JTJ[pt_loc:pt_loc+3, pt_loc:pt_loc+3] += np.dot(pt_derivu, pt_derivu.T)
    
    return JTJ, JTe


def errors(solution):
    points = solution['points']
    poses = solution['poses']
    observations = solution['observations']

    # for both x, y
    error = np.zeros(2 * len(observations))
    for idx, camera_point in enumerate(observations):
        projected_point, point_prime = projection(solution, camera_point[0], camera_point[1])
        # x is % 2 and y is % 2 + 1
        error[2 * idx] = camera_point[2] - projected_point[0]
        error[2 * idx + 1] = camera_point[3] - projected_point[1]

    return error


# Equations (1.2 - 1.8) in notes 1 or Equation 2 in the assignment
def projection(solution, camera_id, point_id):
    poses = solution['poses'][camera_id]
    world_point = solution['points'][point_id]

    # P' = RP + t
    point_prime = np.dot(poses[0:3, 0:3], world_point) + poses[0:3, 3]
    
    # return  f * [X'/Z', Y'/Z'], P' 
    return solution['focal_lengths'][camera_id] * point_prime[:2] / point_prime[2], point_prime


def update_params(solution, update, n_cameras, n_points):
    if solution['is_calibrated']:
        num_variables = 6
    else:
        num_variables = 7

    for i in range(n_cameras):
        curr_index = num_variables*i

        t_update = np.array(update[curr_index+3:curr_index+6])
        psi = np.array(update[curr_index:curr_index+3])
        psi = np.expand_dims(psi, axis=1)
        if not solution['is_calibrated']:
            f_update = np.array(update[curr_index+6])

        # Rodrigues' Formula
        theta = np.linalg.norm(psi)+0.000001
        phi = psi / theta
        sin = np.sin(theta)
        cos = np.cos(theta)
        R_update = cos * np.eye(3) + (1-cos) * np.dot(phi, phi.T) + sin * cross_product(phi)

        solution['poses'][i][0:3, 3] = solution['poses'][i][0:3, 3] + t_update
        solution['poses'][i][0:3, 0:3] = np.dot(solution['poses'][i][0:3, 0:3], R_update)
        if not solution['is_calibrated']:
            solution['focal_lengths'][i] += f_update

    # update all the points
    for i in range(0, n_points):
        curr_index = num_variables*n_cameras + 3*i
        pt_update = update[curr_index:curr_index+3]
        solution['points'][i] = solution['points'][i] + pt_update


# Notes 2 - Equation 1.7
def cross_product(P):
    Phat = [[0, -P[2], P[1]], [P[2], 0, -P[0]], [-P[1], P[0], 0]]
    return np.array(Phat)


def solve_ba_problem(problem):
    '''
    Solves the bundle adjustment problem defined by "problem" dict

    Input:
        problem: bundle adjustment problem containing the following fields:
            - is_calibrated: boolean, whether or not the problem is calibrated
            - observations: list of (cam_id, point_id, x, y)
            - points: [n_points,3] numpy array of 3d points
            - poses: [n_cameras,3,4] numpy array of camera extrinsics - Rotation is 3x3 and translation is 3x1
            - focal_lengths: [n_cameras] numpy array of focal lengths
    Output:
        solution: dictionary containing the problem, with the following fields updated
            - poses: [n_cameras,3,4] numpy array of optmized camera extrinsics
            - points: [n_points,3] numpy array of optimized 3d points
            - (if is_calibrated==False) then focal lengths should be optimized too
                focal_lengths: [n_cameras] numpy array with optimized focal focal_lengths

    Your implementation should optimize over the following variables to minimize reprojection error
        - problem['poses']
        - problem['points']
        - problem['focal_lengths']: if (is_calibrated==False)

    '''

    solution = problem
    # YOUR CODE STARTS
    
    calibrated = solution['is_calibrated']
    observations = solution['observations']
    points = solution['points']
    poses = solution['poses']
    if not calibrated:
        f_lengths = solution['focal_lengths']
    n_cameras = solution['poses'].shape[0]
    n_points = solution['points'].shape[0]
    n_observations = len(solution['observations'])

    lambd = 1
    delta = 1e-4
    loss_op = sys.maxsize
    itr = 0

    while True:
        error = errors(solution)
        norm = 0
        for i in range(error.shape[0]):
            norm += np.linalg.norm(error[i:i+2])

        new_loss = np.linalg.norm(error)
        if loss_op < new_loss:
            solution = previous_solution
            lambd *= 2
        else:
            lambd /= 2

        A, b = LM(solution, error, n_cameras, n_points, n_observations)
        I = np.eye(A.shape[0])
        A = A + lambd * I

        update = np.linalg.solve(A, b)

        if np.linalg.norm(new_loss - loss_op) < delta or np.linalg.norm(update) < delta:
            break

        previous_solution = solution
        update_params(solution, update, n_cameras, n_points)
        loss_op = new_loss

    return solution

# Resourcing functions
def set_max_runtime(ms):
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    resource.setrlimit(resource.RLIMIT_CPU, (ms, hard))
    signal.signal(signal.SIGXCPU, exit_function)


def limit_memory(maxbytes):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxbytes, hard))
    signal.signal(signal.SIGXCPU, exit_function)


def exit_function(signo, frame):
    raise SystemExit(1)


if __name__ == '__main__':
    # UNIX use only
    # set_max_runtime(600) # 10 minutes

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', help="config file")
    args = parser.parse_args()

    # limit_memory(1073741824) # 1GB
    problem = pickle.load(open(args.problem, 'rb'))
    solution = solve_ba_problem(problem)

    solution_path = args.problem.replace(".pickle", "-solution.pickle")
    pickle.dump(solution, open(solution_path, "wb"))
