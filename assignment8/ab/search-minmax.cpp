#include "search.h"
#include "board.h"
#include "eval.h"

/**
 * Implementation of MinMax strategy.
 */
class MinimaxStrategy: public SearchStrategy
{
 public:
    // Defines the name of the strategy
    MinimaxStrategy(): SearchStrategy("Minimax") {}

    // Factory method: just return a new instance of this class
    SearchStrategy* clone() { return new MinimaxStrategy(); }

 private:
    /**
     * Implementation of the strategy.
     */
    void searchBestMove();

    // Minimax algorithm
    int minimax(int depth, bool maximizingPlayer);
};

void MinimaxStrategy::searchBestMove()
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
        eval = minimax(_maxDepth - 1, false);
        takeBack();

        if (eval > bestEval) {
            bestEval = eval;
            bestMove = m;
        }
    }

    foundBestMove(0, bestMove, bestEval);
    finishedNode(0,0);
}

int MinimaxStrategy::minimax(int depth, bool maximizingPlayer)
{
    if (depth == 0) {
        return evaluate();
    }

    int bestEval;
    if (maximizingPlayer) {
        bestEval = minEvaluation();
        Move m;
        MoveList list;
        generateMoves(list);

        while(list.getNext(m)) {
            playMove(m);
            int eval = minimax(depth - 1, false);
            takeBack();

            if (eval > bestEval) {
                bestEval = eval;
            }
        }
    } else {
        bestEval = maxEvaluation();
        Move m;
        MoveList list;
        generateMoves(list);

        while(list.getNext(m)) {
            playMove(m);
            int eval = minimax(depth - 1, true);
            takeBack();

            if (eval < bestEval) {
                bestEval = eval;
            }
        }
    }

    return bestEval;
}

// register ourselve as a search strategy
MinimaxStrategy minimaxStrategy;
