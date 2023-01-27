import pickle
import numpy as np
import argparse
import networkx as nx

# new imports
import itertools
# import signal # UNIX only
# import os # UNIX only
# import resource # UNIX only


# Utility functions
def calculate_Z_brute_force(G,  all_possible_assignments):
    Z = 0

    for assignment in all_possible_assignments:
        unary_potential, binary_potential = 1, 1

        for v in G.nodes:
            unary_potential *= G.nodes[v]['unary_potential'][assignment[v]]
        for u, v in G.edges:
            binary_potential *= G.edges[u, v]['binary_potential'][assignment[u]][assignment[v]]

        Z += (unary_potential * binary_potential)

    return Z


def calculate_joint_probability_brute_force(G, assignment):
    unary_potential, binary_potential = 1, 1

    for v in G.nodes:
        unary_potential *= G.nodes[v]['unary_potential'][assignment[v]]
    for u, v in G.edges:
        binary_potential *= G.edges[u, v]['binary_potential'][assignment[u]][assignment[v]]

    return unary_potential * binary_potential


def calculate_binary_gradient_brute_force(G, Z):
    K = G.graph['K']
    for u in G.nodes:
        for v in G.neighbors(u):
            # already visited
            if v >= u : 
                assignment_switch = np.zeros((K, K))
                assignment_switch[G.nodes[u]['assignment']][G.nodes[v]['assignment']] = 1

                # to optimize storage, `G.edges[u,v]['gradient_binary_potential']` contains the joint probability for that assignment.
                G.edges[u,v]['gradient_binary_potential'] = (assignment_switch - G.edges[u,v]['gradient_binary_potential'])  / G.edges[u,v]['binary_potential']


def calculate_unary_gradient(G):
    for v in G.nodes:
        for domain_value in range(G.graph['K']):
            if domain_value == G.nodes[v]['assignment']:
                G.nodes[v]['gradient_unary_potential'][domain_value] = (1 - G.nodes[v]['marginal_prob'][domain_value]) / G.nodes[v]['unary_potential'][domain_value]
            else:
                G.nodes[v]['gradient_unary_potential'][domain_value] = (-G.nodes[v]['marginal_prob'][domain_value]) / G.nodes[v]['unary_potential'][domain_value]


def upward_belief_propagation(G, node, parent):

    # leaf nodes
    if G.degree[node] == 1 and G.neighbors(node) == parent:
        return G.nodes[node]['unary_potential']

    final_message = G.nodes[node]['unary_potential']
    for neighbor in G.neighbors(node):
        # do not need the message to include node it is being passed to
        if neighbor != parent: 
            binary_potential = G.edges[neighbor, node]['binary_potential']
            # make sure lower vertices are in axis 0
            if neighbor < node: 
                binary_potential = binary_potential.T

            # normalize
            G.edges[neighbor, node]['up_msg'] = np.matmul(binary_potential, upward_belief_propagation(G, neighbor, node))
            final_message = final_message * (G.edges[neighbor, node]['up_msg'] / sum(G.edges[neighbor, node]['up_msg']))

    return final_message


def downward_belief_propagation(G, node, parent):

    for neighbor in G.neighbors(node):
        # do not need the message to include node it is being passed to
        if neighbor != parent: 
            G.edges[neighbor, node]['down_msg'] = G.nodes[node]['unary_potential']
            binary_potential = G.edges[neighbor, node]['binary_potential']
            # always keep lower indexed node in axis 0
            if neighbor > node : 
                binary_potential = binary_potential.T

            for other_neigbor in G.neighbors(node):
                if neighbor != other_neigbor: 
                    if other_neigbor != parent:
                        G.edges[neighbor, node]['down_msg'] = G.edges[neighbor, node]['down_msg'] * G.edges[node, other_neigbor]['up_msg']
                    else:
                        G.edges[neighbor, node]['down_msg'] = G.edges[neighbor, node]['down_msg'] * G.edges[node, other_neigbor]['down_msg']

            # normalize
            G.edges[neighbor, node]['down_msg'] = np.matmul(binary_potential, G.edges[neighbor, node]['down_msg']) 
            G.edges[neighbor, node]['down_msg'] = G.edges[neighbor, node]['down_msg'] / np.sum(G.edges[neighbor, node]['down_msg'])

            downward_belief_propagation(G, neighbor, node)


def calculate_map(G, node, parent):

    def calculate_message_at_each_node(G, node, parent):
        K = G.graph['K']
        if G.degree[node] == 1 and G.neighbors(node) == parent:
            return G.nodes[node]['unary_potential']

        message = G.nodes[node]['unary_potential']
        for neighbor in G.neighbors(node):
            # do not need the message to include node it is being passed to
            if neighbor != parent:
                binary_potential = G.edges[neighbor, node]['binary_potential']
                # always keep lower indexed node in axis 0
                if neighbor < node: 
                    binary_potential = binary_potential.T

                messages_till_neighbor = calculate_message_at_each_node(G, neighbor, node)
                message_at_node = np.zeros(K)

                # calculate m(u -> v) for all u, K
                for assignment_v in range(K):
                    for assignment_u in range(K):
                        new_msg = binary_potential[assignment_v][assignment_u] * messages_till_neighbor[assignment_u]
                        if new_msg > message_at_node[assignment_v]:
                            message_at_node[assignment_v] = new_msg
                            G.edges[neighbor, node]['max_K'][assignment_v] = assignment_u
                            
                # normalize message
                message = message * (message_at_node / sum(message_at_node))

        return message

    def propagate_map(G, node, parent):
        for neighbor in G.neighbors(node):
            if neighbor != parent: 
                node_assignment = int(G.graph['v_map'][node])
                G.graph['v_map'][neighbor] = int(G.edges[node, neighbor]['max_K'][node_assignment])
                propagate_map(G, neighbor, node)

    root_message = calculate_message_at_each_node(G, 0, -1)
    G.graph['v_map'][0] = int(np.argmax(root_message))
    # convert into integer np array
    G.graph['v_map'] = G.graph['v_map'].astype(int)
    propagate_map(G, 0, -1)


def calculate_marginal_belief_propagation(G, node, parent):
    G.nodes[node]['marginal_prob'] = G.nodes[node]['unary_potential']
    for neighbor in G.neighbors(node):
        if neighbor != parent:
            message = G.edges[node, neighbor]['up_msg']
        else:
            message = G.edges[node, neighbor]['down_msg']
        G.nodes[node]['marginal_prob'] = G.nodes[node]['marginal_prob'] * message

    for neighbor in G.neighbors(node):
        if neighbor != parent:
            calculate_marginal_belief_propagation(G, neighbor, node)


def calculate_binary_gradient(G, node, parent):
    K = G.graph['K']
    for neighbor in G.neighbors(node):
        if neighbor != parent:
            joint_probability = G.nodes[node]['unary_potential'][:, None] * G.nodes[neighbor]['unary_potential'][None, :] * G.edges[(node, neighbor)]['binary_potential']
            # keep lower number nodes in axis 0
            if node > neighbor: 
                joint_probability = joint_probability.T

            message_at_neighbor, message_at_node = np.ones(K), np.ones(K)
            for other_neigbor in G.neighbors(neighbor):
                if other_neigbor != node: 
                    message_at_neighbor = message_at_neighbor * G.edges[neighbor, other_neigbor]['up_msg']
            for other_neigbor in G.neighbors(node):
                if neighbor != other_neigbor:
                    if other_neigbor != parent:
                        message_at_node = message_at_node * G.edges[node, other_neigbor]['up_msg']
                    else:
                        message_at_node = message_at_node * G.edges[node, other_neigbor]['down_msg']
                        
            final_message = message_at_node[:, None] * message_at_neighbor[None, :]
            # make sure lower number nodes are at axis 0
            if node > neighbor: 
                final_message = final_message.T
            joint_probability = (joint_probability * final_message) / np.sum(joint_probability * final_message)

            assignment_switch = np.zeros((K, K))
            assignment_switch[G.nodes[node]['assignment']][ G.nodes[neighbor]['assignment']] = 1

            G.edges[node, neighbor]['gradient_binary_potential'] = (assignment_switch - joint_probability) / G.edges[node, neighbor]['binary_potential']

            calculate_binary_gradient(G, neighbor, node)


def print_function(G, Z = None):
    for v in G.nodes:
        print('Marginal Probability: ', G.nodes[v]['marginal_prob'])
    for v in G.nodes:
        print('Unary Gradient:', G.nodes[v]['gradient_unary_potential'])
    for e in G.edges:
        print('Binary Gradient:', G.edges[e]['gradient_binary_potential'])
    print('MAP:', G.graph['v_map'])
    if Z:
        print('Z:', Z)
    for e in G.edges:
        print(e)
        print('Upward messages:', G.edges[e]['up_msg'])

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


def load_graph(filename):
    'load the graphical model (DO NOT MODIFY)'
    return pickle.load(open(filename, 'rb'))


def inference_brute_force(G):
    '''
    Perform probabilistic inference on graph G, and compute the gradients of the log-likelihood
    Input: 
        G: 
            a Graph object in the NetworkX library (version 2.2, https://networkx.github.io/)
        K: 
            G.graph['K']
        unary potentials: 
            G.nodes[v]['unary_potential'], a 1-d numpy array of length K
        binary potentials: 
            G.edges[u, v]['binary_potential'], a 2-d numpy array of shape K x K
        assignment for computing the gradients: 
            G.nodes[v]['assignment'], an integer within [0, K - 1]
    Output:
        G.nodes[v]['marginal_prob']: 
            the marginal probability distribution for v, a 1-d numpy array of length K
        G.graph['v_map']: 
            the MAP assignment, a 1-d numpy arrary of length n, where n is the number of vertices
        G.nodes[v]['gradient_unary_potential']: 
            the gradient of the log-likelihood w.r.t. the unary potential of vetext v, a 1-d numpy array of length K
        G.edges[u, v]['gradient_binary_potential']: 
            the gradient of the log-likelihood w.r.t. the binary potential of edge (u, v), a 2-d numpy array of shape K x K
    '''
    # initialize the output buffers
    for v in G.nodes:
        G.nodes[v]['marginal_prob'] = np.zeros(G.graph['K'])
        G.nodes[v]['gradient_unary_potential'] = np.zeros(G.graph['K'])
    for e in G.edges:
        G.edges[e]['gradient_binary_potential'] = np.zeros((G.graph['K'], G.graph['K']))
    G.graph['v_map'] = np.zeros(len(G.nodes))

    # YOUR CODE STARTS    
    value_domain = range(G.graph['K'])
    all_possible_assignments = list(itertools.product(value_domain, repeat=len(G.nodes))) # cartesian product of `value_domain` duplicated `repeat` times (number of nodes)

    max_probability = 0
    Z = 0
    for assignment in all_possible_assignments:
        joint_probability = calculate_joint_probability_brute_force(G, assignment)
        Z += joint_probability

        if max_probability < joint_probability :
            max_probability = joint_probability
            G.graph['v_map'] = np.array(assignment)

        for v in G.nodes:
            G.nodes[v]['marginal_prob'][assignment[v]] += joint_probability
            # update value of `G.edges[(u, v)]['gradient_binary_potential']` with joint probabilities to optimize storage
            for u in G.neighbors(v):
                # do not visit same edge again
                if u >= v: 
                    G.edges[(u, v)]['gradient_binary_potential'][assignment[v]][assignment[u]] += joint_probability

    # Normalize
    for v in G.nodes:
        G.nodes[v]['marginal_prob'] = G.nodes[v]['marginal_prob'] / sum(G.nodes[v]['marginal_prob'])
    for e in G.edges:
        G.edges[e]['gradient_binary_potential'] = G.edges[e]['gradient_binary_potential'] / np.sum(G.edges[e]['gradient_binary_potential'])
        
    calculate_unary_gradient(G)
    calculate_binary_gradient_brute_force(G, Z)
    print_function(G, Z)

def inference(G):
    '''
    Perform probabilistic inference on graph G, and compute the gradients of the log-likelihood
    Input: 
        G: 
            a Graph object in the NetworkX library (version 2.2, https://networkx.github.io/)
        K: 
            G.graph['K']
        unary potentials: 
            G.nodes[v]['unary_potential'], a 1-d numpy array of length K
        binary potentials: 
            G.edges[u, v]['binary_potential'], a 2-d numpy array of shape K x K
        assignment for computing the gradients: 
            G.nodes[v]['assignment'], an integer within [0, K - 1]
    Output:
        G.nodes[v]['marginal_prob']: 
            the marginal probability distribution for v, a 1-d numpy array of length K
        G.graph['v_map']: 
            the MAP assignment, a 1-d numpy arrary of length n, where n is the number of vertices
        G.nodes[v]['gradient_unary_potential']: 
            the gradient of the log-likelihood w.r.t. the unary potential of vetext v, a 1-d numpy array of length K
        G.edges[u, v]['gradient_binary_potential']: 
            the gradient of the log-likelihood w.r.t. the binary potential of edge (u, v), a 2-d numpy array of shape K x K
    '''
    # initialize the output buffers
    for v in G.nodes:
        G.nodes[v]['marginal_prob'] = np.zeros(G.graph['K'])
        G.nodes[v]['gradient_unary_potential'] = np.zeros(G.graph['K'])
    for e in G.edges:
        G.edges[e]['gradient_binary_potential'] = np.zeros((G.graph['K'], G.graph['K']))
    G.graph['v_map'] = np.zeros(len(G.nodes))
    
    # YOUR CODE STARTS
    for e in G.edges:
        G.edges[e]['up_msg'] = np.zeros(G.graph['K'])
        G.edges[e]['down_msg'] = np.zeros(G.graph['K'])

    upward_belief_propagation(G, 0, -1)
    downward_belief_propagation(G, 0, -1)
    calculate_marginal_belief_propagation(G, 0, -1)

    for v in G.nodes:
        G.nodes[v]['marginal_prob'] = G.nodes[v]['marginal_prob'] / sum(G.nodes[v]['marginal_prob'])
    for e in G.edges:
        G.edges[e]['max_K'] = np.zeros(G.graph['K'])

    calculate_map(G, 0, -1)
    calculate_unary_gradient(G)
    calculate_binary_gradient(G, 0, -1)

    print_function(G)

if __name__ == '__main__':
    # UNIX use only
    # set_max_runtime(54000) # 20 minutes
    # limit_memory(2147483648) # 2GB

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='The input graph')
    args = parser.parse_args()
    G = load_graph(args.input)
    # inference_brute_force(G)
    inference(G)
    pickle.dump(G, open('results_' + args.input, 'wb'))