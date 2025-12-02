import java.util.List;
import java.util.Map;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.io.IOException;

/**
 * CardRebalanceAnalyzer analyzes card statistics and flags cards that may need a rebalance.
 * Usage: Call analyze(cards, cardStats, totalGames) after simulation.
 */
public class CardRebalanceAnalyzer {
    public static void analyze(List<SPCard> allGameCards, Map<String, CardBalanceSimulator.CardStats> cardStatsMap, int totalGamesRun) {
        printAnalysis(allGameCards, cardStatsMap, totalGamesRun, null);
    }

    public static void analyzeToFile(List<SPCard> allGameCards, Map<String, CardBalanceSimulator.CardStats> cardStatsMap, int totalGamesRun, String filename) {
        printAnalysis(allGameCards, cardStatsMap, totalGamesRun, filename);
    }

    private static void printAnalysis(List<SPCard> allGameCards, Map<String, CardBalanceSimulator.CardStats> cardStatsMap, int totalGamesRun, String filename) {
        PrintWriter out = null;
        try {
            if (filename != null) {
                out = new PrintWriter(new FileWriter(filename));
            }
            PrintWriter pw = out != null ? out : new PrintWriter(System.out);
            pw.println("\n=== Card Rebalance Analysis ===");
            pw.println("Total Games Run: " + totalGamesRun);
            pw.println("-----------------------------------------------------------------------------------");
            pw.println("Card Name                  | Purchases/Game | Win Rate (When Purchased) | Sample Size | Rebalance Suggestion");
            pw.println("-----------------------------------------------------------------------------------");
            for (SPCard card : allGameCards) {
                CardBalanceSimulator.CardStats stats = cardStatsMap.getOrDefault(card.name, new CardBalanceSimulator.CardStats());
                double purchasesPerGame = totalGamesRun > 0 ? (double) stats.totalTimesPurchased / totalGamesRun : 0.0;
                double winRate = stats.totalTimesPurchased > 0 ? (double) stats.timesPurchasedByWinner / stats.totalTimesPurchased : 0.0;
                String suggestion = getRebalanceSuggestion(purchasesPerGame, winRate, stats.totalTimesPurchased);
                pw.printf("%-26s | %-13.3f | %-24.2f%% | %-10d | %s\n",
                    card.name + " (" + card.cost + ")",
                    purchasesPerGame,
                    winRate * 100,
                    stats.totalTimesPurchased,
                    suggestion
                );
            }
            pw.println("-----------------------------------------------------------------------------------\n");
            if (out != null) {
                out.close();
            }
        } catch (IOException e) {
            System.err.println("Could not write analysis file: " + e.getMessage());
        }
    }

    private static String getRebalanceSuggestion(double purchasesPerGame, double winRate, int sampleSize) {
        if (sampleSize < 100) {
            return "INSUFFICIENT DATA";
        }
        if (winRate > 0.55) {
            if (purchasesPerGame < 0.05) {
                return "TOO STRONG BUT RARE (Consider accessibility nerf)";
            }
            return "OVERPOWERED (Suggest Nerf)";
        } else if (winRate < 0.45) {
            if (purchasesPerGame > 0.2) {
                return "POPULAR BUT WEAK (Consider buff)";
            }
            return "UNDERPOWERED (Suggest Buff)";
        } else if (purchasesPerGame < 0.01) {
            return "IGNORED (Consider making more attractive)";
        }
        return "BALANCED";
    }
}
