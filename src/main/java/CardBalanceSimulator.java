import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.Comparator;

/**
 * CardBalanceSimulator
 *
 * Runs many games of Saint Petersburg and builds card-balance statistics:
 *  - total buys
 *  - wins when bought
 *  - winrate if bought
 *  - games where the card appeared
 *  - recommendation (BUFF / NERF / OK) based on winrate
 *
 * Integrates directly with:
 *  - SPSimulateGame
 *  - SPGameTranscript
 *  - SPBuyAction
 *  - SPAction
 *  - SPPlayer
 */
public class CardBalanceSimulator {

    /** Statistics record for a card. */
    static class CardStats {
        final String name;
        int buys = 0;
        int winsWhenBought = 0;
        int gamesSeen = 0;

        CardStats(String name) { this.name = name; }
    }

    // Map from card name to stats
    private final Map<String, CardStats> stats = new HashMap<>();

    /**
     * Run N simulated games using the provided players.
     */
    public void runSimulations(int numGames, SPPlayer... players) {
        for (int g = 0; g < numGames; g++) {

            SPGameTranscript transcript = SPSimulateGame.simulateGame(players);
            processTranscript(transcript);

            // Progress message
            if ((g + 1) % 100 == 0 || g == numGames - 1) {
                System.out.printf("Completed %d / %d games%n", g + 1, numGames);
            }
        }
    }

    /**
     * Process a single game transcript to update our card statistics.
     */
    private void processTranscript(SPGameTranscript transcript) {
        List<String> playerNames = transcript.getPlayerNames();
        int numPlayers = playerNames.size();

        // Track purchases per player for this game
        ArrayList<ArrayList<String>> boughtByPlayer = new ArrayList<>();
        for (int i = 0; i < numPlayers; i++) {
            boughtByPlayer.add(new ArrayList<>());
        }

        // Count buys
        for (SPAction action : transcript.getActions()) {
            if (action instanceof SPBuyAction) {
                SPBuyAction buy = (SPBuyAction) action;
                String cardName = buy.card.name;

                CardStats cs = stats.computeIfAbsent(cardName, CardStats::new);
                cs.buys++;

                // record buyer
                if (buy.player >= 0 && buy.player < numPlayers) {
                    boughtByPlayer.get(buy.player).add(cardName);
                }
            }
        }

        // GamesSeen: record cards seen in this game
        java.util.HashSet<String> seenThisGame = new java.util.HashSet<>();
        for (ArrayList<String> purchases : boughtByPlayer) {
            for (String name : purchases) {
                if (!seenThisGame.contains(name)) {
                    seenThisGame.add(name);
                    stats.computeIfAbsent(name, CardStats::new).gamesSeen++;
                }
            }
        }

        // WinsWhenBought: if winner bought a card, increment it
        boolean[] winners = transcript.getWinners();
        for (int p = 0; p < winners.length; p++) {
            if (winners[p]) {
                for (String name : boughtByPlayer.get(p)) {
                    stats.computeIfAbsent(name, CardStats::new).winsWhenBought++;
                }
            }
        }
    }

    /**
     * Recommendation based on winrate:
     *   < 0.50 → BUFF
     *   > 0.53 → NERF
     *   else  → OK
     */
    private String balanceRecommendation(double winRate) {
        if (winRate < 0.50) return "BUFF";
        if (winRate > 0.65) return "NERF";
        return "BALANCED";
    }

    /**
     * Print results to console.
     */
    public void printResults() {
        List<CardStats> list = new ArrayList<>(stats.values());
        list.sort(Comparator.comparingInt((CardStats cs) -> cs.buys).reversed());

        System.out.printf(
            "%-30s  %8s  %12s  %12s  %10s  %14s%n",
            "Card", "Buys", "WinsWhenBought", "WinRateIfBought", "GamesSeen", "Recommendation"
        );

        for (CardStats cs : list) {
            double winRate = cs.buys > 0 ? (double) cs.winsWhenBought / cs.buys : 0.0;
            String rec = balanceRecommendation(winRate);

            System.out.printf(
                "%-30s  %8d  %12d  %11.3f  %10d  %14s%n",
                cs.name, cs.buys, cs.winsWhenBought, winRate, cs.gamesSeen, rec
            );
        }
    }

    /**
     * Write CSV file.
     */
    public void writeCsv(String filename) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(filename))) {
            pw.println("Card,Buys,WinsWhenBought,WinRateIfBought,GamesSeen,Recommendation");

            for (CardStats cs : stats.values()) {
                double winRate = cs.buys > 0 ? (double) cs.winsWhenBought / cs.buys : 0.0;
                String rec = balanceRecommendation(winRate);

                String safeName = cs.name.replace(",", ""); // ensure CSV is clean
                pw.printf("%s,%d,%d,%.6f,%d,%s%n",
                        safeName, cs.buys, cs.winsWhenBought, winRate, cs.gamesSeen, rec);
            }
        }
    }

    /**
     * A simple random player (uniform random legal action).
     * Lets you run the simulator without needing advanced AI players.
     */
    public static class SPRandomNoHandPlayer extends SPPlayer {
        public SPRandomNoHandPlayer() { super("Random_Player"); }

        @Override
        public int getAction(SPState state) {
            ArrayList<SPAction> legal = state.getLegalActions();
            if (legal.isEmpty()) return -1;
            return SPAction.cardRandom.nextInt(legal.size());
        }
    }

    /**
     * Example main: runs 500 simulated games with two random players.
     * Replace the players with your desired bots (e.g., OKMCTSBasedPlayer).
     */
    public static void main(String[] args) {
        int games = 10000;
        if (args.length >= 1) {
            try { games = Integer.parseInt(args[0]); } catch (Exception ignored) {}
        }

        // Replace these with your bots if desired
        SPPlayer p1 = new SPRandomNoHandPlayer();
        SPPlayer p2 = new SPRandomNoHandPlayer();

        CardBalanceSimulator sim = new CardBalanceSimulator();
        sim.runSimulations(games, p1, p2);

        System.out.println("\nFinal Card Balance Statistics:\n");
        sim.printResults();

        try {
            sim.writeCsv("card_balance_results.csv");
            System.out.println("\nWrote CSV to card_balance_results.csv");
        } catch (IOException e) {
            System.err.println("Failed to write CSV: " + e.getMessage());
        }
    }
}
