import numpy as np
import pickle
import argparse

# new imports
import sys
import copy

np.set_printoptions(threshold=sys.maxsize, suppress=True)

# resourcing
import resource
import signal
import tracemalloc

# Uses equation 1.7 in the notes part 3 (LM equation without rescaling)
def LM(solution, error, n_cameras, n_points, n_observations, lambd):

    if solution['is_calibrated']:
        num_variables = 6
    else:
        num_variables = 7

    J_x = np.array([])
    J_p = np.array([])
    b = np.array([])
    F = np.array([])
    d = np.array([])

    for index, camera_point in enumerate(solution['observations']):
        camera_id = camera_point[0]
        point_id = camera_point[1]
        f = solution['focal_lengths'][camera_id]

        if camera_id == 0:
            J_r1 = np.zeros((2, num_variables * n_cameras))
            J_r2 = np.zeros((2, 3 * n_points))

        cam_loc = num_variables * camera_id
        pt_loc = 3 * point_id

        P = solution['points'][point_id]
        R = solution['poses'][camera_id]

        projected_point, point_prime = projection(solution, camera_point[0], camera_point[1])

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

        J_r1[0, cam_loc:cam_loc+3] = dudpsi
        J_r1[0, cam_loc+3:cam_loc+6] = dudt
        if not solution['is_calibrated']:
            J_r1[0, cam_loc+6] = dudf
            
        J_r1[1, cam_loc:cam_loc+3] = dvdpsi
        J_r1[1, cam_loc+3:cam_loc+6] = dvdt
        if not solution['is_calibrated']:
            J_r1[1, cam_loc+6] = dvdf

        J_r2[0, pt_loc:pt_loc+3] = dudP
        J_r2[1, pt_loc:pt_loc+3] = dvdP

        J_p = np.hstack([J_p.T, J_r1.T]).T if J_p.size else J_r1
        J_x = np.hstack([J_x.T, J_r2.T]).T if J_x.size else J_r2

        F = np.hstack([F.T, projected_point.T]) if F.size else projected_point.T
        b_elem = np.array([camera_point[2], camera_point[3]])
        b = np.hstack([b.T, b_elem.T]) if b.size else b_elem.T

    return J_p, J_x, b, F


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

    new_loss = 0
    for idx in range(error.shape[0]):
        new_loss += np.linalg.norm(error[idx:idx+2])
    new_loss = new_loss / error.shape[0]

    return new_loss


# Equations (1.2 - 1.8) in notes 1 or Equation 2 in the assignment
def projection(solution, camera_id, point_id):
    poses = solution['poses'][camera_id]
    world_point = solution['points'][point_id]

    # P' = RP + t
    point_prime = np.dot(poses[0:3, 0:3], world_point) + poses[0:3, 3]
    
    # return  f * [X'/Z', Y'/Z'], P' 
    return solution['focal_lengths'][camera_id] * point_prime[:2] / point_prime[2], point_prime


def update_params(solution, delta_p, delta_x, n_cameras, n_points):
    if solution['is_calibrated']:
        num_variables = 6
    else:
        num_variables = 7

    # update extrinsics
    for i in range(n_cameras):
        curr_index = num_variables*i

        t_update = np.array(delta_p[curr_index+3:curr_index+6])
        psi = np.array(delta_p[curr_index:curr_index+3])
        psi = np.expand_dims(psi, axis=1)
        if not solution['is_calibrated']:
            f_update = np.array(delta_p[curr_index+6])

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

    # update points
    for i in range(n_points):
        curr_index = 3*i
        pt_update = delta_x[curr_index:curr_index+3]
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

    solution['observations'] = sorted(solution['observations'], key=lambda x: (x[1], x[0]))
    
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
    # itr = 0
    previous_solution = None

    while True:
        error = errors(solution)
        # print(error)
        new_loss = error
        if loss_op > new_loss:
            if previous_solution:
                solution = previous_solution 
            else:
                 solution = problem
            lambd *= 2
        else:
            lambd /= 2

        J_p, J_x, b, f = LM(solution, error, n_cameras, n_points, n_observations, lambd)

        A = np.dot(J_p.T, J_p) 
        A +=  lambd * np.eye(A.shape[0])
        B = np.dot(J_p.T, J_x)
        D = np.dot(J_x.T, J_x) 
        D +=  lambd * np.eye(D.shape[0])
        D_inv = np.linalg.inv(D)
        e_p = np.dot(J_p.T, (b-f))
        e_x = np.dot(J_x.T, (b-f))      

        JTJ = np.block([[A, B], [B.T, D]])

        delta_p =  np.dot(np.linalg.inv(A - np.dot(np.dot(B, D_inv), B.T)), (e_p - np.dot(np.dot(B, D_inv), e_x)))
        delta_x = np.dot(D_inv, (e_x - np.dot(B.T, delta_p)))

        if np.linalg.norm(new_loss - loss_op) < delta:
            break

        previous_solution = solution
        update_params(solution, delta_p, delta_x, n_cameras, n_points)
        loss_op = new_loss

        # itr += 1
        # print(itr)
    
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
    set_max_runtime(600) # 10 minutes

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', help="config file")
    args = parser.parse_args()

    problem = pickle.load(open(args.problem, 'rb'))
    
    # UNIX only
    limit_memory(1073741824) # 1GB

    solution = solve_ba_problem(problem)

    solution_path = args.problem.replace(".pickle", "-solution.pickle")
    pickle.dump(solution, open(solution_path, "wb"))
