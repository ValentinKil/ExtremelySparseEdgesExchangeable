import numpy as np 
from scipy.sparse import coo_matrix
from numba import njit

def edgesexchageablegraph(weights, t_step):
    N_max = weights.shape[0]
    data = []
    row = []
    col = []

    for i in range(N_max):
        for j in range(i + 1, N_max):
            prob = weights[i] * weights[j]
            val = np.random.binomial(t_step, prob)
            if val > 0:
                # Add both (i,j) and (j,i) for symmetry
                row.extend([i, j])
                col.extend([j, i])
                data.extend([val, val])
    
    M_sparse = coo_matrix((data, (row, col)), shape=(N_max, N_max))
    return M_sparse


@njit(nogil=True)
def _numba_generate_edges(weights, t_step):
    """
    A Numba-jitted helper function to generate graph edges.
    This function contains the core logic that can be compiled to fast machine code.
    """
    N_max = weights.shape[0]
    # Use Python lists to accumulate results, as they are efficiently
    # handled by Numba and can be converted to NumPy arrays upon return.
    rows, cols, data = [], [], []

    # Numba excels at optimizing simple nested loops like this.
    for i in range(N_max):
        for j in range(i + 1, N_max):
            prob = weights[i] * weights[j]
            # np.random.binomial is supported directly by Numba.
            val = np.random.binomial(t_step, prob)
            if val > 0:
                # Add symmetric edges (i,j) and (j,i)
                rows.append(i)
                cols.append(j)
                data.append(val)

                rows.append(j)
                cols.append(i)
                data.append(val)

    # Numba will efficiently convert these lists of scalars to NumPy arrays.
    return (
        np.array(rows, dtype=np.int64),
        np.array(cols, dtype=np.int64),
        np.array(data, dtype=np.int64),
    )

def numba_edgesexchageablegraph(weights, t_step):
    """
    Generates a sparse graph using a Numba-accelerated helper function.
    This is expected to be the fastest version for large N.
    """
    N_max = weights.shape[0]
    row, col, data = _numba_generate_edges(weights, t_step)
    return coo_matrix((data, (row, col)), shape=(N_max, N_max))


def graphstatistics(M):
    '''Calculate the number of edges and the average degree of the graph.'''
    active_nodes = np.unique(np.concatenate([M.row, M.col]))
    num_nodes = len(active_nodes)
    
    num_multi_edges = M.data.sum() // 2
    
    upper = (M.row < M.col)
    num_single_edges = np.count_nonzero(upper)  # one per undirected edge

    return num_nodes, num_multi_edges, num_single_edges