#include "search.h"
#include "board.h"
#include "eval.h"
#include <algorithm> // Include this header for std::min and std::max


/**
 * Implementation of Alpha-Beta strategy.
 */
class AlphaBetaStrategy: public SearchStrategy
{
public:
    // Defines the name of the strategy
    AlphaBetaStrategy(): SearchStrategy("AlphaBeta") {}

    // Factory method: just return a new instance of this class
    SearchStrategy* clone() { return new AlphaBetaStrategy(); }

private:

    /**
     * Implementation of the strategy.
     */
    void searchBestMove();

    // Alpha-Beta algorithm
    int alphaBeta(int depth, int alpha, int beta, bool maximizingPlayer);
};

void AlphaBetaStrategy::searchBestMove()
{
    int bestEval = minEvaluation();
    int eval;
    Move bestMove;
    Move m;
    MoveList list;

    // generate list of allowed moves, put them into <list>
    generateMoves(list);

    // loop over all moves
    while(list.getNext(m)) {
        playMove(m);
        eval = alphaBeta(_maxDepth - 1, minEvaluation(), maxEvaluation(), false);
        takeBack();
        
        if (eval > bestEval) {
            bestEval = eval;
            bestMove = m;
        }
    }

    foundBestMove(0, bestMove, bestEval);
    finishedNode(0, 0);
}

int AlphaBetaStrategy::alphaBeta(int depth, int alpha, int beta, bool maximizingPlayer)
{
    if (depth == 0) {
        return evaluate();
    }

    Move m;
    MoveList list;
    generateMoves(list);

    if (maximizingPlayer) {
        int maxEval = minEvaluation();
        while (list.getNext(m)) {
            playMove(m);
            int eval = alphaBeta(depth - 1, alpha, beta, false);
            takeBack();
            maxEval = std::max(maxEval, eval);
            alpha = std::max(alpha, eval);
            if (beta <= alpha) {
                break;
            }
        }
        return maxEval;
    } else {
        int minEval = maxEvaluation();
        while (list.getNext(m)) {
            playMove(m);
            int eval = alphaBeta(depth - 1, alpha, beta, true);
            takeBack();
            minEval = std::min(minEval, eval);
            beta = std::min(beta, eval);
            if (beta <= alpha) {
                break;
            }
        }
        return minEval;
    }
}

// register ourselve as a search strategy
AlphaBetaStrategy alphaBetaStrategy;
