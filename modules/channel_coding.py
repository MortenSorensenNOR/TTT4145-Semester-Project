"""Channel coding enums and helpers."""

from enum import Enum
from dataclasses import dataclass
import numpy as np
import numba
from numba import njit


@njit(cache=True)
def _check_node_update_numba(
    L_v2c: np.ndarray,
    L_c2v: np.ndarray,
    check_edges_padded: np.ndarray,
    check_degrees: np.ndarray,
    alpha: float,
) -> None:
    """Numba-accelerated check node update (normalized min-sum)."""
    num_checks = len(check_degrees)

    for i in range(num_checks):
        degree = check_degrees[i]
        edges = check_edges_padded[i, :degree]

        # Compute sign product and find two smallest magnitudes
        sign_product = 1
        min1 = np.inf  # smallest magnitude
        min2 = np.inf  # second smallest magnitude
        min1_idx = 0

        for idx in range(degree):
            e = edges[idx]
            msg = L_v2c[e]
            if msg < 0:
                sign_product *= -1
            mag = abs(msg)
            if mag < min1:
                min2 = min1
                min1 = mag
                min1_idx = idx
            elif mag < min2:
                min2 = mag

        # Compute outgoing messages
        for idx in range(degree):
            e = edges[idx]
            msg = L_v2c[e]

            # Sign: product excluding this edge
            out_sign = sign_product
            if msg < 0:
                out_sign *= -1  # undo this edge's contribution

            # Magnitude: min excluding this edge
            if idx == min1_idx:
                out_mag = min2
            else:
                out_mag = min1

            L_c2v[e] = alpha * out_sign * out_mag


@njit(cache=True)
def _variable_node_update_numba(
    L_v2c: np.ndarray,
    L_c2v: np.ndarray,
    llr_channel: np.ndarray,
    var_edges_padded: np.ndarray,
    var_degrees: np.ndarray,
    edge_vars: np.ndarray,
) -> None:
    """Numba-accelerated variable node update."""
    num_vars = len(var_degrees)

    for j in range(num_vars):
        degree = var_degrees[j]
        edges = var_edges_padded[j, :degree]

        # Sum of all incoming check messages
        total = llr_channel[j]
        for idx in range(degree):
            total += L_c2v[edges[idx]]

        # Outgoing messages (exclude each edge's incoming)
        for idx in range(degree):
            e = edges[idx]
            L_v2c[e] = total - L_c2v[e]


@njit(cache=True)
def _compute_beliefs_numba(
    L_c2v: np.ndarray,
    llr_channel: np.ndarray,
    var_edges_padded: np.ndarray,
    var_degrees: np.ndarray,
    L_total: np.ndarray,
) -> None:
    """Compute total beliefs for hard decision."""
    num_vars = len(var_degrees)

    for j in range(num_vars):
        degree = var_degrees[j]
        total = llr_channel[j]
        for idx in range(degree):
            total += L_c2v[var_edges_padded[j, idx]]
        L_total[j] = total

class CodeRates(Enum):
    """Supported channel coding rates."""
    HALF_RATE = 1
    THREE_QUARTER_RATE = 2

class BaseMatrix:
    """Base matrix for generating parity matrix"""
    base_matrix = np.array([
        [ 0,   -1,   -1,   -1,    0,    0,   -1,   -1,    0,   -1,   -1,    0,    1,    0,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1],
        [22,    0,   -1,   -1,   17,   -1,    0,    0,   12,   -1,   -1,   -1,   -1,    0,    0,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1],
        [ 6,   -1,    0,   -1,   10,   -1,   -1,   -1,   24,   -1,    0,   -1,   -1,   -1,    0,    0,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1],
        [ 2,   -1,   -1,    0,   20,   -1,   -1,   -1,   25,    0,   -1,   -1,   -1,   -1,   -1,    0,    0,   -1,   -1,   -1,   -1,   -1,   -1,   -1],
        [23,   -1,   -1,   -1,    3,   -1,   -1,   -1,    0,   -1,    9,   11,   -1,   -1,   -1,   -1,    0,    0,   -1,   -1,   -1,   -1,   -1,   -1],
        [24,   -1,   23,    1,   17,   -1,    3,   -1,   10,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    0,    0,   -1,   -1,   -1,   -1,   -1],
        [25,   -1,   -1,   -1,    8,   -1,   -1,   -1,    7,   18,   -1,   -1,    0,   -1,   -1,   -1,   -1,   -1,    0,    0,   -1,   -1,   -1,   -1],
        [13,   24,   -1,   -1,    0,   -1,    8,   -1,    6,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    0,    0,   -1,   -1,   -1],
        [ 7,   20,   -1,   16,   22,   10,   -1,   -1,   23,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    0,    0,   -1,   -1],
        [11,   -1,   -1,   -1,   19,   -1,   -1,   -1,   13,   -1,    3,   17,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    0,    0,   -1],
        [25,   -1,    8,   -1,   23,   18,   -1,   14,    9,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    0,    0],
        [ 3,   -1,   -1,   -1,   16,   -1,   -1,    2,   25,    5,   -1,   -1,    1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    0]
    ], dtype=int)

@dataclass
class LDPCConfig:
    n: int # codeword length
    k: int # message length
    Z: int # circulant size (27 for n=648)
    code_rate: CodeRates

class LDPC:
    """Placeholder LDPC codec implementation."""

    def __init__(self, config: LDPCConfig) -> None:
        """Initialize the LDPC codec placeholder."""
        self.config = config
        self.n = config.n
        self.k = config.k
        self.Z = config.Z
        self.code_rate = config.code_rate
        if self.code_rate == CodeRates.HALF_RATE:
            assert(2 * self.k == self.n)
        elif self.code_rate == CodeRates.THREE_QUARTER_RATE:
            assert(self.k / self.n == 0.75)

        assert self.n == 648 and self.Z == 27 and self.code_rate == CodeRates.HALF_RATE, \
            "Only n = 648 with Z=27 with half rate coding is implemented for now"

        self.H = self._expand_h()
        self.check_neighbors, self.var_neighbors = self._build_adj_list(self.H)

        # Pre-compute edge structures for fast decoding
        self._build_edge_structures()

    def circulant_multiply(self, vector, shift):
        """Multiply a Z-length vector by a cyclically shifted identity matrix.

        H[k, (k+shift) % Z] = 1 means result[k] = v[(k+shift) % Z],
        which is equivalent to np.roll(v, -shift).
        """
        if shift == -1:
            return np.zeros(self.Z, dtype=int)
        return np.roll(vector, -shift)

    def encode(self, message):
        """Encode k-bit message to a n-bit codeword

        message: length k (324) bit array
        returns: length n (648) codeword [systemic | parity]
        """
        assert len(message) == self.k
        assert self.n == 648
        assert self.Z == 27

        H_base = BaseMatrix.base_matrix
        num_parity_blocks = H_base.shape[0]
        num_systemic_blocks = H_base.shape[1] - num_parity_blocks

        s_blocks = message.reshape(num_systemic_blocks, self.Z)
        p_blocks = np.zeros((num_parity_blocks, self.Z), dtype=int)

        # compute partial sum of A[i, j] * s[j] for each row i
        lambda_sums = np.zeros((num_parity_blocks, self.Z), dtype=int)
        for i in range(num_parity_blocks):
            for j in range(num_systemic_blocks):
                shift = H_base[i, j]
                if shift != -1:
                    lambda_sums[i] ^= self.circulant_multiply(s_blocks[j], shift)

        # compute first parity block p[0]
        # 802.11 structure needs to accumulate all lambda_sums
        p0_sum = np.zeros(self.Z, dtype=int)
        for i in range(num_parity_blocks):
            p0_sum ^= lambda_sums[i]
        p_blocks[0] = p0_sum

        # compute remaining parity blocks using back-substitutions
        # p[1] from row 0: λ[0] + P^1 * p[0] + P^0 * p[1] = 0
        # The shift for p[0] in row 0 is 1 (from base_matrix[0, 12] = 1)
        p_blocks[1] = lambda_sums[0] ^ self.circulant_multiply(p_blocks[0], 1)

        # p[2] to p[6] from rows 1-5 (standard staircase)
        for i in range(2, 7):
            p_blocks[i] = lambda_sums[i-1] ^ p_blocks[i-1]

        # p[7] from row 6: row 6 has p[0] (col 12), p[6] (col 18), p[7] (col 19)
        # λ[6] + p[0] + p[6] + p[7] = 0 => p[7] = λ[6] + p[0] + p[6]
        p_blocks[7] = lambda_sums[6] ^ p_blocks[0] ^ p_blocks[6]

        # p[8] to p[11] from rows 7-10 (standard staircase)
        for i in range(8, num_parity_blocks):
            p_blocks[i] = lambda_sums[i-1] ^ p_blocks[i-1]

        # combine systematic and parity
        codeword = np.concatenate([s_blocks.flatten(), p_blocks.flatten()])
        return codeword

    def decode(self, llr_channel: np.ndarray, max_iterations: int = 50) -> np.ndarray:
        """Decode using min-sum belief propagation"""
        assert len(llr_channel) == self.n
        hard_decision = np.zeros(self.n, dtype=int)

        num_checks = self.H.shape[0]
        num_vars   = self.H.shape[1]

        # initialize messages
        # L_v2c[i][j] = message from variable j to check i
        # L_c2v[i][j] = message from check i to variable j
        L_v2c = {}
        L_c2v = {}

        # initialize variable-to-check messages with channel LLRs
        for j in range(num_vars):
            for i in self.var_neighbors[j]:
                L_v2c[(i, j)] = llr_channel[j]

        # initialize check-to-variable messages to zero
        for i in range(num_checks):
            for j in self.check_neighbors[i]:
                L_c2v[(i, j)] = 0.0

        # belief propagation iterations
        for iteration in range(max_iterations):
            # check node update (min-sum)
            for i in range(num_checks):
                neighbors = self.check_neighbors[i]
                
                for j in neighbors:
                    # compute message to j using all other neighbors
                    other_neighbors = [jj for jj in neighbors if jj != j]

                    if len(other_neighbors) == 0:
                        L_c2v[(i, j)] = 0.0
                        continue
                    
                    # min-sum - product of signs * mimimum magnitude
                    sign = 1
                    min_mag = float('inf')

                    for jj in other_neighbors:
                        msg = L_v2c[(i, jj)]
                        sign *= -1 if msg < 0 else 1
                        mag = abs(msg)
                        min_mag = mag if mag < min_mag else min_mag

                    L_c2v[(i, j)] = sign * min_mag

            # variable node update
            for j in range(num_vars):
                neighbors = self.var_neighbors[j]
                for i in neighbors:
                    # sum channel llr and all other incoming check messages
                    total = llr_channel[j]
                    for ii in neighbors:
                        if ii != i:
                            total += L_c2v[(ii, j)]
                    L_v2c[(i, j)] = total

            # compute hard decision
            L_total = np.zeros(num_vars)
            for j in range(num_vars):
                L_total[j] = llr_channel[j]
                for i in self.var_neighbors[j]:
                    L_total[j] += L_c2v[(i, j)]

            hard_decision = (L_total < 0).astype(int)

            # check if valid codeword
            syndrome = self.H @ hard_decision % 2
            if np.all(syndrome == 0):
                return hard_decision[:self.k] # valid codeword -> extract systemic bits

        # max iterations reached - return best guess
        return hard_decision[:self.k]

    def _build_edge_structures(self):
        """Pre-compute edge-based data structures for fast decoding.

        Instead of using dictionaries keyed by (check, var) tuples, we use
        flat arrays indexed by edge number. This enables vectorized operations.
        """
        # Count edges and build edge list
        edges = []
        for i, neighbors in enumerate(self.check_neighbors):
            for j in neighbors:
                edges.append((i, j))

        self.num_edges = len(edges)
        self.edge_checks = np.array([e[0] for e in edges], dtype=np.int32)
        self.edge_vars = np.array([e[1] for e in edges], dtype=np.int32)

        # Build mapping from (check, var) to edge index
        self.edge_idx = {}
        for e, (i, j) in enumerate(edges):
            self.edge_idx[(i, j)] = e

        # For each check node, list of edge indices connected to it
        self.check_edge_indices = [[] for _ in range(len(self.check_neighbors))]
        for e, (i, j) in enumerate(edges):
            self.check_edge_indices[i].append(e)

        # For each variable node, list of edge indices connected to it
        self.var_edge_indices = [[] for _ in range(len(self.var_neighbors))]
        for e, (i, j) in enumerate(edges):
            self.var_edge_indices[j].append(e)

        # Convert to arrays with padding for vectorization
        max_check_degree = max(len(x) for x in self.check_edge_indices)
        max_var_degree = max(len(x) for x in self.var_edge_indices)

        self.check_edges_padded = np.full(
            (len(self.check_neighbors), max_check_degree), -1, dtype=np.int32
        )
        self.check_degrees = np.array(
            [len(x) for x in self.check_edge_indices], dtype=np.int32
        )
        for i, edges_list in enumerate(self.check_edge_indices):
            self.check_edges_padded[i, :len(edges_list)] = edges_list

        self.var_edges_padded = np.full(
            (len(self.var_neighbors), max_var_degree), -1, dtype=np.int32
        )
        self.var_degrees = np.array(
            [len(x) for x in self.var_edge_indices], dtype=np.int32
        )
        for j, edges_list in enumerate(self.var_edge_indices):
            self.var_edges_padded[j, :len(edges_list)] = edges_list

    def decode_fast(
        self,
        llr_channel: np.ndarray,
        max_iterations: int = 50,
        alpha: float = 0.75,
    ) -> np.ndarray:
        """Decode using normalized min-sum belief propagation (optimized).

        This is a faster implementation using NumPy arrays instead of dictionaries.

        Args:
            llr_channel: Channel LLRs (length n)
            max_iterations: Maximum BP iterations
            alpha: Normalization factor for min-sum (0.75-0.875 typical).
                   Set to 1.0 for standard min-sum without normalization.

        Returns:
            Decoded message bits (length k)
        """
        assert len(llr_channel) == self.n
        num_checks = self.H.shape[0]
        num_vars = self.H.shape[1]

        # Message arrays indexed by edge
        L_v2c = np.zeros(self.num_edges, dtype=np.float64)
        L_c2v = np.zeros(self.num_edges, dtype=np.float64)

        # Initialize v2c messages with channel LLRs
        L_v2c[:] = llr_channel[self.edge_vars]

        # Belief propagation iterations
        for iteration in range(max_iterations):
            # === Check node update (normalized min-sum) ===
            for i in range(num_checks):
                edge_indices = self.check_edges_padded[i, :self.check_degrees[i]]

                # Get incoming messages
                incoming = L_v2c[edge_indices]

                # Compute sign and magnitude
                signs = np.sign(incoming)
                signs[signs == 0] = 1  # treat zero as positive
                mags = np.abs(incoming)

                # Product of all signs
                sign_product = np.prod(signs)

                # For each outgoing edge, compute message
                for idx, e in enumerate(edge_indices):
                    # Exclude current edge from computation
                    other_mask = np.ones(len(edge_indices), dtype=bool)
                    other_mask[idx] = False

                    # Sign: product of other signs
                    out_sign = sign_product * signs[idx]  # divide out current sign

                    # Magnitude: minimum of other magnitudes
                    out_mag = np.min(mags[other_mask]) if np.any(other_mask) else 0.0

                    # Normalized min-sum
                    L_c2v[e] = alpha * out_sign * out_mag

            # === Variable node update ===
            for j in range(num_vars):
                edge_indices = self.var_edges_padded[j, :self.var_degrees[j]]

                # Sum of all incoming check messages
                total_incoming = np.sum(L_c2v[edge_indices])

                # For each outgoing edge, exclude that edge's incoming message
                for e in edge_indices:
                    L_v2c[e] = llr_channel[j] + total_incoming - L_c2v[e]

            # === Compute hard decision and check syndrome ===
            L_total = llr_channel.copy()
            for j in range(num_vars):
                edge_indices = self.var_edges_padded[j, :self.var_degrees[j]]
                L_total[j] += np.sum(L_c2v[edge_indices])

            hard_decision = (L_total < 0).astype(np.int32)

            # Check if valid codeword
            syndrome = self.H @ hard_decision % 2
            if np.all(syndrome == 0):
                return hard_decision[:self.k]

        # Max iterations reached
        return hard_decision[:self.k]

    def decode_numba(
        self,
        llr_channel: np.ndarray,
        max_iterations: int = 50,
        alpha: float = 0.75,
    ) -> np.ndarray:
        """Decode using normalized min-sum with Numba acceleration.

        This is significantly faster than the pure Python implementations.
        First call has JIT compilation overhead, subsequent calls are fast.

        Args:
            llr_channel: Channel LLRs (length n)
            max_iterations: Maximum BP iterations
            alpha: Normalization factor for min-sum (0.75-0.875 typical).
                   Set to 1.0 for standard min-sum without normalization.

        Returns:
            Decoded message bits (length k)
        """
        assert len(llr_channel) == self.n

        llr_channel = llr_channel.astype(np.float64)

        # Message arrays
        L_v2c = llr_channel[self.edge_vars].copy()
        L_c2v = np.zeros(self.num_edges, dtype=np.float64)
        L_total = np.zeros(self.n, dtype=np.float64)

        for iteration in range(max_iterations):
            # Check node update
            _check_node_update_numba(
                L_v2c, L_c2v,
                self.check_edges_padded, self.check_degrees,
                alpha
            )

            # Variable node update
            _variable_node_update_numba(
                L_v2c, L_c2v,
                llr_channel,
                self.var_edges_padded, self.var_degrees,
                self.edge_vars
            )

            # Compute beliefs and hard decision
            _compute_beliefs_numba(
                L_c2v, llr_channel,
                self.var_edges_padded, self.var_degrees,
                L_total
            )

            hard_decision = (L_total < 0).astype(np.int32)

            # Check syndrome
            syndrome = self.H @ hard_decision % 2
            if np.all(syndrome == 0):
                return hard_decision[:self.k]

        return hard_decision[:self.k]

    def _expand_h(self) -> np.ndarray:
        """Expand base matrix to full H matrix"""
        H_base = BaseMatrix.base_matrix
        num_block_rows = H_base.shape[0]
        num_block_cols = H_base.shape[1]

        H = np.zeros((num_block_rows * self.Z, num_block_cols * self.Z), dtype=int)

        for i in range(num_block_rows):
            for j in range(num_block_cols):
                shift = H_base[i, j]
                if shift != -1:
                    for k in range(self.Z):
                        H[i * self.Z + k, j * self.Z + (k + shift) % self.Z] = 1

        return H

    def _build_adj_list(self, H: np.ndarray):
        """Build adjecency list from H matrix"""
        num_chekcs, num_vars = H.shape

        check_neighbors = [[] for _ in range(num_chekcs)] # list of vars
        var_neighbors   = [[] for _ in range(num_vars)]   # list of checks
        for i in range(num_chekcs):
            for j in range(num_vars):
                if H[i, j] == 1:
                    check_neighbors[i].append(j)
                    var_neighbors[j].append(i)
        return check_neighbors, var_neighbors
