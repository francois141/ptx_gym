from triton_ptx.evaluation import EvaluatedCandidate


def select_winner(results) -> EvaluatedCandidate:
    return sorted(results)[0]
