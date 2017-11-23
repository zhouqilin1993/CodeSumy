from src.opennmt.onmt import CNNEncoder, CNNDecoder
from src.opennmt.onmt import ContextGateFactory
from src.opennmt.onmt import ConvMultiStepAttention
from src.opennmt.onmt import CopyGenerator, CopyGeneratorLossCompute
from src.opennmt.onmt import Embeddings
from src.opennmt.onmt import GlobalAttention
from src.opennmt.onmt import ImageEncoder
from src.opennmt.onmt import LayerNorm, Bottle, BottleLinear, \
    BottleLayerNorm, BottleSoftmax, Elementwise
from src.opennmt.onmt import MatrixTree
from src.opennmt.onmt import MultiHeadedAttention
from src.opennmt.onmt import StackedLSTM, StackedGRU
from src.opennmt.onmt import TransformerEncoder, TransformerDecoder
from src.opennmt.onmt import WeightNormConv2d
from src.opennmt.onmt import check_sru_requirement

can_use_sru = check_sru_requirement()
if can_use_sru:
    from src.opennmt.onmt import SRU


# For flake8 compatibility.
__all__ = [GlobalAttention, ImageEncoder, CopyGenerator, MultiHeadedAttention,
           LayerNorm, Bottle, BottleLinear, BottleLayerNorm, BottleSoftmax,
           TransformerEncoder, TransformerDecoder, Embeddings, Elementwise,
           MatrixTree, WeightNormConv2d, ConvMultiStepAttention,
           CNNEncoder, CNNDecoder, StackedLSTM, StackedGRU, ContextGateFactory,
           CopyGeneratorLossCompute]

if can_use_sru:
    __all__.extend([SRU, check_sru_requirement])
