# DecomposableAttention
## Introduction
A pytorch implementation of the Decomposable Attention model (Parikh et al., 2016).
The model is used for a Natural Language Inference (NLI) task. It expects input in the form of (premises, hypotheses) and returns the unnormalized scores of the classes Neutral(N), Entailment(E) and Contradiction(C).

A slight deviation from the original paper is the inclusion of positional embedding in the _Encoder_ class, as described in the transformer paper(Vaswani et al., 2017).

## References
[1] Parikh, A. P., Täckström, O., Das, D., & Uszkoreit, J. (2016). A Decomposable Attention Model for Natural Language Inference (arXiv:1606.01933). arXiv. http://arxiv.org/abs/1606.01933

[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need (arXiv:1706.03762). arXiv. https://doi.org/10.48550/arXiv.1706.03762
