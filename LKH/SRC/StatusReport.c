#include "LKH.h"

void StatusReport(GainType Cost, double EntryTime, char *Suffix)
{
    if (Penalty) {
        printff("Cost = " GainFormat "_" GainFormat, CurrentPenalty, Cost);
        if (Optimum != MINUS_INFINITY && Optimum != 0) {
            if (!Penalty ||
                (ProblemType != CCVRP &&
                 ProblemType != CBTSP &&
                 ProblemType != CBnTSP &&
                 ProblemType != GCTSP &&
                 ProblemType != CCCTSP &&
                 ProblemType != KTSP &&
                 ProblemType != MLP &&
                 ProblemType != PTSP &&
                 ProblemType != TRP &&
                 Penalty != Penalty_MTSP_MINMAX &&
                 Penalty != Penalty_MTSP_MINMAX_SIZE))
                printff(", Gap = %0.4f%%",
                        100.0 * (Cost - Optimum) / Optimum);
            else
                printff(", Gap = %0.4f%%",
                        100.0 * (CurrentPenalty - Optimum) / Optimum);
        }
        printff(", Time = %0.2f sec. %s",
                fabs(GetTime() - EntryTime), Suffix);
    } else {
        printff("Cost = " GainFormat, Cost);
        if (Optimum != MINUS_INFINITY && Optimum != 0)
            printff(", Gap = %0.4f%%", 100.0 * (Cost - Optimum) / Optimum);
        printff(", Time = %0.2f sec.%s%s",
                fabs(GetTime() - EntryTime), Suffix,
                Cost < Optimum ? "<" : Cost == Optimum ? " =" : "");
    }
    printff("\n");
}
