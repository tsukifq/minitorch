from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    
    y2 = f(*[vals[i] + epsilon if i == arg else vals[i] for i in range(len(vals))])
    y1 = f(*[vals[i] - epsilon if i == arg else vals[i] for i in range(len(vals))])
    return (y2 - y1) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    Visited = set()
    order = []
    def dfs(v: Variable) -> None:
        if v.unique_id in Visited:
            return
        Visited.add(v.unique_id)
        for p in v.parents:
            dfs(p)
        if not v.is_constant():
            order.append(v)
    dfs(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    queue = topological_sort(variable)
    
    # Dictionary to store derivatives for intermediate nodes.
    # We map variable unique_id -> current derivative value.
    derivatives = {}
    
    # Initialize the derivative for the starting node (root).
    derivatives[variable.unique_id] = deriv

    # Traverse in reverse topological order (Root -> Leaves)
    for v in reversed(queue):
        # retrieved the derivative for the current node v
        d_v = derivatives.get(v.unique_id)
        
        # If this node is a leaf, we are done with it, just accumulate.
        if v.is_leaf():
            v.accumulate_derivative(d_v)
        else:
            # If it is an intermediate node, use chain rule to propagate to parents.
            # v.chain_rule(d_v) returns pairs of (parent, d_parent_contribution)
            for parent, d_p in v.chain_rule(d_v):
                # Accumulate the contribution to the parent's derivative.
                if parent.unique_id in derivatives:
                    derivatives[parent.unique_id] += d_p
                else:
                    derivatives[parent.unique_id] = d_p


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
