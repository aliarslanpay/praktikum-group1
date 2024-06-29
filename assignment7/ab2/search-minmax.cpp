#include <omp.h>
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
    int minimax(int depth, bool maximizingPlayer, Board board);
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
    #pragma omp parallel for schedule(runtime) shared(bestEval, bestMove) firstprivate(m, eval)
    for(int i = 0; i < list.getLength(); i++) {
        Board boardCopy = *_board; // Make a copy of the board
        list.getNext(m);
        boardCopy.playMove(m);
        eval = minimax(_maxDepth - 1, false, boardCopy); // Pass the copy
        boardCopy.takeBack();

        #pragma omp critical
        {
            if (eval > bestEval) {
                bestEval = eval;
                bestMove = m;
            }
        }
    }

    foundBestMove(0, bestMove, bestEval);
    finishedNode(0,0);
}

int MinimaxStrategy::minimax(int depth, bool maximizingPlayer, Board board)
{
    if (depth == 0) {
        return evaluate();
    }

    int bestEval;
    if (maximizingPlayer) {
        bestEval = minEvaluation();
        Move m;
        MoveList list;
        board.generateMoves(list);

        while(list.getNext(m)) {
            board.playMove(m);
            int eval = minimax(depth - 1, false, board);
            board.takeBack();

            if (eval > bestEval) {
                bestEval = eval;
            }
        }
    } else {
        bestEval = maxEvaluation();
        Move m;
        MoveList list;
        board.generateMoves(list);

        while(list.getNext(m)) {
            board.playMove(m);
            int eval = minimax(depth - 1, true, board);
            board.takeBack();

            if (eval < bestEval) {
                bestEval = eval;
            }
        }
    }

    return bestEval;
}

// register ourselves as a search strategy
MinimaxStrategy minimaxStrategy;