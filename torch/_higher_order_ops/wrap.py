import inspect
import logging

import torch
from torch._ops import HigherOrderOperator
from torch.utils.checkpoint import checkpoint, uid

log = logging.getLogger(__name__)



# Used for testing the HigherOrderOperator mechanism
class Wrap(HigherOrderOperator):
    def __init__(self):
        super().__init__("wrap")

    def __call__(self, func, *args, **kwargs):
        # Dynamo already traces the body of HigherOrderOp beforehand when it
        # so no need to trace into it.
        import torch._dynamo  # noqa: F401
        from torch._dynamo import disable

        @disable
        def wrapper():
            result = func(*args, **kwargs)
            return result

        return wrapper()

wrap = Wrap()

class HandleActivationCheckpoint(HigherOrderOperator):
    """
    This operator is supposed to be used only with torch.compile stack.
    There are two modes in this operator:

    Mode 1: WrapActivationCheckpoint
    This mode is used for selective checkpointing + torch.compile.
    Under this mode, we wrap torch.utils.checkpoint. This avoids
    TorchDynamo to look into saved tensor hooks and directly passes the control
    to AOT Autograd, which is ok with tracing saved tensor hooks. As a result of
    AOT tracing torch.utils.checkpoint code, we have a backward graph with
    recomputed forward nodes.

    However, we might deprecate this mode soon. The difficulty arises in the
    functionalization of rng ops. Today, there are two different
    functionalization of rng ops - one at AOT autograd and other at Inductor.
    And they are difficult to map to each other. The rng states also complicate
    pattern matching in Inductor. Due to the ease of implementation, we are
    currently inclined towards functionalization at Inductor level, which means
    that duplication/recomputation is done as a compiler pass in the
    partitioners. See TagActivationCheckpoint for more information.

    Mode 2: TagActivationCheckpoint
    This mode is used for full activation checkpointing + torch.compile.
    Under this mode, the operator accepts a Fx graph module which needs to be checkpointed.
    This operator adds "recomputable" tag to the nodes of the Fx graph that
    should be recomputed.

    The goal is to avoid both Dynamo and AOT Autograd to trace through saved
    tensor hooks, and rather rely on the partitioners to actually duplicate the
    nodes. This sits well in the torch.compile stack, because by the time graph
    reaches partitioner, inductor has already run its functionalization of rng
    ops. Therefore, the duplication of nodes, by design, respects the rng states
    in the forward and recomputed forward in backward.
    """

    def __init__(self):
        super().__init__("handle_activation_checkpoint")
        self.context_fn = None

    @staticmethod
    def divide_kwargs(kwargs):
        """
        checkpoint fn can have mixed kwargs between checkpointed fn and
        checkpoint fn itself. For example
        >> def gn(x, y, z=None):
        >>     a = torch.matmul(x, y)
        >>     if z is not None:
        >>         return torch.matmul(a, z)
        >>     return a
        >> def fn(x, y, z):
        >>     return torch.cos(checkpoint(gn, x, y, use_reentrant=False, z=z))
        In the above case, z belongs to checkpointed function gn, but
        use_reentrant belongs to the checkpoint function. This function splits
        the kwargs into checkpoint_kwargs and gmod_kwargs (or
        checkpointed_fn_kwargs).
        We do sorting to ensure same graph from run to run for better
        debuggability. It is not required for correctness.
        """
        ckpt_signature = inspect.signature(checkpoint)
        checkpoint_keys = set()
        for name in ckpt_signature.parameters:
            if name in ("function", "args", "kwargs"):
                continue
            checkpoint_keys.add(name)

        # `preserve_rng_state` is not a regular kwarg
        checkpoint_keys.add("preserve_rng_state")

        checkpoint_kwargs = {name: kwargs[name] for name in kwargs.keys() if name in checkpoint_keys}
        gmod_kwargs = {name: kwargs[name] for name in kwargs.keys() if name not in checkpoint_keys}
        return checkpoint_kwargs, gmod_kwargs

    def tag_nodes(self, gmod):
        unique_graph_id = next(uid)
        for node in gmod.graph.nodes:
            if node.op in ("call_function", "call_method", "call_module"):
                node.meta["recompute"] = unique_graph_id
        return gmod

    def __call__(self, gmod, *args, **kwargs):
        import torch.fx.traceback as fx_traceback
        from torch.fx import Interpreter
        # TODO - This is a temporary sitaution where we have two versions of
        # checkpointing implemetation. We will converge on one and remove the other.
        if self.context_fn is not None or torch._functorch.config.functionalize_rng_ops:
            # Mode 1: WrapActivationCheckpoint
            assert kwargs.get("use_reentrant", False) is False, \
                "use_reentrant=True is not supported under torch.compile " + \
                "when selective checkpointing is used " + \
                "or when `torch._functorch.config.functionalize_rng_ops` is True."
            assert kwargs.get("preserve_rng_state", False) is False, \
                "preserve_rng_state=True is not supported under torch.compile " + \
                "when selective checkpointing is used " + \
                "or when `torch._functorch.config.functionalize_rng_ops` is True. " + \
                "If your program relies on this to have deterministic output in the checkpointed region, " + \
                "it means the checkpointed region has ops that use random number generator (e.g. torch.dropout). " + \
                "Please remove those random ops as they are not supported under selective checkpointing + torch.compile."

            log.warning("""
Detected selective checkpointing is used under torch.compile.
Please make sure the checkpointed region does not contain:

1. random ops (e.g. torch.dropout)
2. in-place ops (e.g. torch.relu_)
""")

            kwargs["use_reentrant"] = False
            kwargs["preserve_rng_state"] = False
            kwargs["context_fn"] = self.context_fn
            # We first tag all nodes as "recompute" in this graph, and then we undo the "recompute" tag
            # for specific nodes in user-defined CachedTorchDispatchMode.
            gmod = self.tag_nodes(gmod)
            # Using interpreter allows preservation of metadata through torch.compile stack.
            with fx_traceback.preserve_node_meta():
                ret = checkpoint(Interpreter(gmod).run, *args, **kwargs)
                return ret
        else:
            # Mode 2: TagActivationCheckpoint
            gmod = self.tag_nodes(gmod)
            # Using interpreter allows preservation of metadata through torch.compile stack.
            with fx_traceback.preserve_node_meta():
                return Interpreter(gmod).run(*args)

handle_activation_checkpoint = HandleActivationCheckpoint()
