#include "LKH.h"
  
/* The GCTSP_InitialTour function computes an initial tour for a
 * general colored TSP.
 */

GainType GCTSP_InitialTour()
{
    Node *N;
    GainType Cost;
    int Set;
    double EntryTime = GetTime();

    if (TraceLevel >= 1)
        printff("GCTSP = ");
    assert(ProblemType == GCTSP);
    assert(!Asymmetric);
    int OldTraceLevel = TraceLevel;
    TraceLevel = 0;
    InitialTourAlgorithm = GREEDY;
    GreedyTour();
    InitialTourAlgorithm = GCTSP_ALG;
    TraceLevel = OldTraceLevel;
    for (Set = 2; Set <= Salesmen; Set++)
        Follow(&NodeSet[Dim + Set - 1], &NodeSet[Dim + Set - 2]);
    N = FirstNode;
    do
        N->OldSuc = N->Suc;
    while ((N = N->Suc) != FirstNode);

    for (Set = 1; Set <= Salesmen; Set++) {
        N = FirstNode;
        do {
            if (N->Id < Dim && ColorAllowed[Set][N->Id])
                Follow(N, &NodeSet[Dim + Set - 1]);
        } while ((N = N->OldSuc) != FirstNode);
    }
    Cost = 0;
    N = FirstNode;
    do
        Cost += C(N, N->Suc) - N->Pi - N->Suc->Pi;
    while ((N = N->Suc) != FirstNode);
    Cost /= Precision;
    CurrentPenalty = PLUS_INFINITY;
    CurrentPenalty = Penalty();
    if (TraceLevel >= 1) {
        printff(GainFormat "_" GainFormat, CurrentPenalty, Cost);
        if (Optimum != MINUS_INFINITY && Optimum != 0)
            printff(", Gap = %0.2f%%",
                    100.0 * (CurrentPenalty - Optimum) / Optimum);
        printff(", Time = %0.2f sec.\n", fabs(GetTime() - EntryTime));
    }
    return Cost;
}
