/**
 * A real world, sequential strategy:
 * Alpha/Beta with Iterative Deepening (ABID)
 *
 * (c) 2005, Josef Weidendorfer
 */

#include <omp.h>
#include <stdio.h>
#include "search.h"
#include "board.h"

class ABIDStrategy: public SearchStrategy
{
public:
    ABIDStrategy(): SearchStrategy("ABID", 2) {}
    SearchStrategy* clone() { return new ABIDStrategy(); }

    Move& nextMove() { return _pv[1]; }

private:
    void searchBestMove();
    int alphabeta(int depth, int alpha, int beta, Board &board);

    Variation _pv;
    Move _currentBestMove;
    bool _inPV;
    int _currentMaxDepth;
};

void ABIDStrategy::searchBestMove()
{
    int alpha = -15000, beta = 15000;
    int nalpha, nbeta, currentValue = 0;

    _pv.clear(_maxDepth);
    _currentBestMove.type = Move::none;
    _currentMaxDepth = 1;

    do {
        while (1) {
            nalpha = alpha, nbeta = beta;
            _inPV = (_pv[0].type != Move::none);

            if (_sc && _sc->verbose()) {
                char tmp[100];
                sprintf(tmp, "Alpha/Beta [%d;%d] with max depth %d", alpha, beta, _currentMaxDepth);
                _sc->substart(tmp);
            }

            currentValue = alphabeta(0, alpha, beta, *_board);

            if (currentValue > 14900 || currentValue < -14900)
                _stopSearch = true;

            if (_currentBestMove.type == Move::none)
                _stopSearch = false;

            if (_stopSearch) break;

            if (currentValue <= nalpha) {
                alpha = -15000;
                if (beta < 15000) beta = currentValue + 1;
                continue;
            }
            if (currentValue >= nbeta) {
                if (alpha > -15000) alpha = currentValue - 1;
                beta = 15000;
                continue;
            }
            break;
        }

        alpha = currentValue - 200, beta = currentValue + 200;

        if (_stopSearch) break;

        _currentMaxDepth++;
    } while (_currentMaxDepth <= _maxDepth);

    _bestMove = _currentBestMove;
}

int ABIDStrategy::alphabeta(int depth, int alpha, int beta, Board &board)
{
    int currentValue = -14999 + depth, value;
    Move m;
    MoveList list;
    bool depthPhase, doDepthSearch;

    int maxType = (depth < _currentMaxDepth - 1) ? Move::maxMoveType :
                  (depth < _currentMaxDepth) ? Move::maxPushType :
                  Move::maxOutType;

    board.generateMoves(list);

    if (_sc && _sc->verbose()) {
        char tmp[100];
        sprintf(tmp, "Alpha/Beta [%d;%d], %d moves (%d depth)", alpha, beta,
                list.count(Move::none), list.count(maxType));
        _sc->startedNode(depth, tmp);
    }

    if (_inPV) {
        m = _pv[depth];
        if ((m.type != Move::none) &&
            (!list.isElement(m, 0, true)))
            m.type = Move::none;
        if (m.type == Move::none) _inPV = false;
    }

    depthPhase = true;
    int bestValue = currentValue;
    Move bestMove;
    Variation bestPv;

    int numMoves = list.count(Move::none);
    bool stopFlag = false;

    #pragma omp parallel
    {
        Board threadBoard = board;
        Move threadMove;
        bool threadDepthPhase = depthPhase;

        #pragma omp for schedule(dynamic) nowait
        for (int i = 0; i < numMoves; i++) {
            if (stopFlag) continue;

            bool validMove = false;
            if (threadDepthPhase) {
                #pragma omp critical
                {
                    if (!list.getNext(threadMove, maxType)) {
                        threadDepthPhase = false;
                        validMove = list.getNext(threadMove, Move::none);
                    } else {
                        validMove = true;
                    }
                }
            } else {
                #pragma omp critical
                {
                    validMove = list.getNext(threadMove, Move::none);
                }
            }
            if (!validMove) continue;

            doDepthSearch = threadDepthPhase && (threadMove.type <= maxType);
            threadBoard.playMove(threadMove);

            if (!threadBoard.isValid()) {
                value = 14999 - depth;
            } else {
                if (doDepthSearch) {
                    value = -alphabeta(depth + 1, -beta, -alpha, threadBoard);
                } else {
                    value = evaluate();
                }
            }
            threadBoard.takeBack();

            #pragma omp critical
            {
                if (value > bestValue) {
                    bestValue = value;
                    bestMove = threadMove;
                    bestPv.update(depth, threadMove);
                    if (_sc) _sc->foundBestMove(depth, threadMove, bestValue);
                    if (depth == 0)
                        _currentBestMove = threadMove;
                    if (bestValue > 14900 || bestValue >= beta) {
                        if (_sc) _sc->finishedNode(depth, bestPv.chain(depth));
                        stopFlag = true;
                    }
                    if (bestValue > alpha) alpha = bestValue;
                }
            }
            if (_stopSearch) {
                stopFlag = true;
            }
            threadMove.type = Move::none;
        }
    }

    if (_sc) _sc->finishedNode(depth, bestPv.chain(depth));
    return bestValue;
}

// register ourselve
ABIDStrategy abidStrategy;
