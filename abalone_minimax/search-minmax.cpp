#include "search.h"
#include "board.h"
#include "eval.h"
#include <algorithm>
#include <cstring>
#include <chrono>
#include <omp.h>

#define MAX_EVAL 20000
#define MIN_EVAL -20000
#define INITIAL_REMAINING_TIME 60.0
#define BOARD_SIZE 121
#define NUM_THREADS 16

/**
 * Implementation of MinMax strategy.
 */
class MinimaxStrategy : public SearchStrategy
{
public:
    // Defines the name of the strategy
    MinimaxStrategy() : SearchStrategy("Minimax") {}

    // Factory method: just return a new instance of this class
    SearchStrategy *clone() { return new MinimaxStrategy(); }

private:
    /**
     * Implementation of the strategy.
     */
    int minimax_openmp(int depth, Board board);
    int minimax(int depth, Board *board, Evaluator *evaluator, int alpha, int beta);
    bool compare_boards(const int *curr_board, const int *prev_board);
    void searchBestMove();

    int move_count = 0;
    double remaining_time_secs = INITIAL_REMAINING_TIME;
    int best_evaluation_value;
    int adaptive_search_depth;
    Move prev_move;
    Move prev_2_move;
    int prev_board[BOARD_SIZE];
    int prev_2_board[BOARD_SIZE];
};

bool MinimaxStrategy::compare_boards(const int *curr_board, const int *prev_board)
{
    return std::equal(curr_board, curr_board + BOARD_SIZE, prev_board);
}

int MinimaxStrategy::minimax(int depth, Board *board, Evaluator *evaluator, int alpha, int beta)
{
    bool is_maximizing_player = !(depth % 2);

    if (depth >= adaptive_search_depth)
        return (1 - is_maximizing_player * 2) * evaluator->calcEvaluation(board);

    MoveList list;
    Move move;
    // generate list of allowed moves, put them into <list>
    board->generateMoves(list);

    int current_value = is_maximizing_player  ? MIN_EVAL : MAX_EVAL;

    // loop over all moves
    while (list.getNext(move)) {
        board->playMove(move);
        int eval = minimax(depth + 1, board, evaluator, alpha, beta);
        board->takeBack();

        if (is_maximizing_player) {
            current_value = std::max(current_value, eval);
            alpha = std::max(alpha, eval);
            if (current_value >= beta)
                break;
        }
        else {
            current_value = std::min(current_value, eval);
            beta = std::min(beta, eval);
            if (current_value <= alpha)
                break;
        }
    }

    return current_value;
}

int MinimaxStrategy::minimax_openmp(int depth, Board board)
{
    int best_evaluation_value = MIN_EVAL;

    MoveList list;

    // generate list of allowed moves, put them into <list>
    board.generateMoves(list);
    int total_moves = list.getLength();

#pragma omp parallel for schedule(dynamic, 1) shared(best_evaluation_value) firstprivate(board)
    for (int i = total_moves-1; i >= 0; --i) {
        Move &move = list.move[i];

        int eval;
        Evaluator evaluator;

        board.playMove(move);
        eval = minimax(depth + 1, &board, &evaluator, MIN_EVAL, MAX_EVAL);
        board.takeBack();

        if (eval > best_evaluation_value) {
#pragma omp critical
            {
                best_evaluation_value = eval;
                _bestMove = move;
            }
        }
    }

    return best_evaluation_value;
}

void MinimaxStrategy::searchBestMove()
{
    auto start = std::chrono::high_resolution_clock::now();

    //  If the boards are the same, use the second to last move
    if (move_count >= 2 && compare_boards(_board->fieldArray(), prev_2_board)) {
        _bestMove = prev_2_move;
    }
    else
    {
        // Calculate adaptive search depth based on remaining time
        if (remaining_time_secs < 5) {
            adaptive_search_depth = 3;
        } else if (remaining_time_secs < 10) {
            adaptive_search_depth = 4;
        } else if (remaining_time_secs < 20) {
            adaptive_search_depth = 5;
        } else if (remaining_time_secs < 30) {
            adaptive_search_depth = 5;
        } else {
            adaptive_search_depth = 6;
        }

        omp_set_dynamic(0);
        omp_set_num_threads(NUM_THREADS);
        best_evaluation_value = minimax_openmp(0, *_board);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto time_in_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

    remaining_time_secs -= (time_in_microseconds / 1e6);

    if (move_count >= 1)
    {
        std::memcpy(prev_2_board, prev_board, BOARD_SIZE * sizeof(int));
        std::memcpy(prev_board, _board->fieldArray(), BOARD_SIZE * sizeof(int));
        prev_2_move = prev_move;
        prev_move = _bestMove;
    }

    move_count++;
}

// register ourselve as a search strategy
MinimaxStrategy minimaxStrategy;

