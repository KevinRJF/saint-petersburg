import java.util.ArrayList;
// import java.util.Arrays; // No longer needed
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * CardBalanceSimulator is a utility class designed to run multiple
 * games and record card purchase frequency and win rates for all card types.
 *
 * NOTE: The 'simulateGameAndTrackActions' method is a TEMPLATE and MUST be 
 * replaced with your actual game simulation logic using SPState, SPAction, and SPPlayer.
 */
public class CardBalanceSimulator {

    // --- Configuration ---
    private static final int NUM_GAMES = 10000000; // Increased iterations for statistical relevance
    private static final String OUTPUT_FILENAME = "card_balance_report.txt";

    // Dynamically track all cards in the game using SPCard.ALL_CARDS
    private static List<SPCard> allGameCards = null;

    // Map to store statistics: Key = Card Name, Value = CardStats
    private Map<String, CardStats> cardStatsMap = new HashMap<>();
    private int[] playerWinCounts = new int[2]; // Player 0 and Player 1
    private int totalGamesRun = 0;

    // We need a Random instance for the mock simulation
    private Random random = new Random();

    /**
     * Inner class to hold statistics for a single card.
     */
    public static class CardStats {
        int totalTimesPurchased = 0;
        int timesPurchasedByWinner = 0;
    }

    /**
     * Entry point to run the simulation loop.
     */
    public void runSimulations() {
        // Initialize all known cards in the map to ensure they appear in the report, even if unpurchased.
        if (allGameCards == null) {
            allGameCards = SPCard.ALL_CARDS;
        }
        allGameCards.forEach(card -> cardStatsMap.put(card.name, new CardStats()));
        
        System.out.println("Starting " + NUM_GAMES + " game simulations...");

        // Instantiate players (Assuming two MCTS players are competing)
        SPPlayer player0 = new OKMCTSBasedPlayer(); 
        SPPlayer player1 = new OKMCTSBasedPlayer();
        
        for (int i = 0; i < NUM_GAMES; i++) {
            // Track the game's history of purchases
            Map<Integer, List<String>> purchaseHistory = new HashMap<>();
            purchaseHistory.put(0, new ArrayList<>());
            purchaseHistory.put(1, new ArrayList<>());

            // The simulation returns the ID of the winner (0 or 1) and the purchase history
            int winner = simulateGameAndTrackActions(player0, player1, purchaseHistory);

            if (winner != -1) {
                totalGamesRun++;
                playerWinCounts[winner]++;
                updateCardStatistics(winner, purchaseHistory);
            }

            if ((i + 1) % 1000 == 0) {
                System.out.println("Completed " + (i + 1) + " games...");
            }
        }

        generateReportFile();
        System.out.println("\nSimulation finished. Report generated: " + OUTPUT_FILENAME);
    }

    /**
     * !!! TEMPLATE METHOD: YOU MUST IMPLEMENT THIS !!!
     * Runs a single game and records which cards were purchased by which player.
     * * @param p0 Player 0 implementation.
     * @param p1 Player 1 implementation.
     * @param purchaseHistory A map to store purchases: Key=Player ID, Value=List of Card Names.
     * @return The ID of the winning player (0 or 1), or -1 if the game was a draw or invalid.
     */
    private int simulateGameAndTrackActions(SPPlayer p0, SPPlayer p1, Map<Integer, List<String>> purchaseHistory) {
        // --- START TEMPORARY MOCK IMPLEMENTATION ---
        
        // **IMPORTANT:** Replace this entire block with your actual game simulation logic.
        // This mock code simply simulates random purchases and a biased winner determination.

        // Use the actual card objects for purchases
        List<SPCard> mockCards = allGameCards;
        
        // Mock Game Length (e.g., 8 rounds)
        for (int round = 0; round < 8; round++) {
            for (int player = 0; player < 2; player++) {
                // Mock purchase decision
                if (random.nextDouble() < 0.15) {
                    SPCard purchasedCard;
                    // Bias the selection to mock the MoC being purchased more often than others
                    List<SPCard> mocCards = mockCards.stream()
                        .filter(c -> c.name.toLowerCase().contains("mistress of ceremonies"))
                        .collect(Collectors.toList());
                    if (!mocCards.isEmpty() && random.nextDouble() < 0.10) {
                        purchasedCard = mocCards.get(random.nextInt(mocCards.size()));
                    } else {
                        // Choose a random card other than MoC
                        List<SPCard> otherCards = mockCards.stream()
                            .filter(c -> mocCards.isEmpty() || !mocCards.contains(c))
                            .collect(Collectors.toList());
                        if (!otherCards.isEmpty()) {
                            purchasedCard = otherCards.get(random.nextInt(otherCards.size()));
                        } else {
                            purchasedCard = mockCards.get(0); // Fallback
                        }
                    }
                    purchaseHistory.get(player).add(purchasedCard.name);
                }
            }
        }
        
        // Mock Winner Determination: 
        // A player is highly likely to win if they bought the MoC
        
        // Use actual card names for MoC detection
        boolean p0BoughtMoC = purchaseHistory.get(0).stream().anyMatch(n -> n.toLowerCase().contains("mistress of ceremonies"));
        boolean p1BoughtMoC = purchaseHistory.get(1).stream().anyMatch(n -> n.toLowerCase().contains("mistress of ceremonies"));

        if (p0BoughtMoC && !p1BoughtMoC) return 0; // P0 wins easily
        if (p1BoughtMoC && !p0BoughtMoC) return 1; // P1 wins easily
        
        // If neither or both bought MoC, it's a toss-up (50/50)
        return random.nextDouble() < 0.5 ? 0 : 1;
        
        // --- END TEMPORARY MOCK IMPLEMENTATION ---
    }
    
    /**
     * Updates the global statistics map based on the results of one game.
     * @param winner The ID of the winning player.
     * @param purchaseHistory The purchases made during the game.
     */
    private void updateCardStatistics(int winner, Map<Integer, List<String>> purchaseHistory) {
        for (int player = 0; player < 2; player++) {
            boolean isWinner = (player == winner);
            
            for (String cardName : purchaseHistory.get(player)) {
                // Get or initialize stats for the card
                CardStats stats = cardStatsMap.getOrDefault(cardName, new CardStats());
                cardStatsMap.put(cardName, stats); // Ensure it's in the map if it was dynamically added
                
                stats.totalTimesPurchased++;
                
                if (isWinner) {
                    stats.timesPurchasedByWinner++;
                }
            }
        }
    }

    /**
     * Generates and writes the final report to a file.
     */
    private void generateReportFile() {
        try (PrintWriter out = new PrintWriter(new FileWriter(OUTPUT_FILENAME))) {
            out.println("=== Card Balance Report (" + totalGamesRun + " Simulations) ===");
            out.println("Total Games Run: " + totalGamesRun);
            out.printf("Player 0 Win Rate: %.2f%%\n", (double) playerWinCounts[0] / totalGamesRun * 100);
            out.printf("Player 1 Win Rate: %.2f%%\n", (double) playerWinCounts[1] / totalGamesRun * 100);
            out.println("-----------------------------------------------------------------------------------");
            out.println("Card Name                  | Total Purchases | Purchased by Winner | Win Rate (When Purchased) | Conclusion");
            out.println("-----------------------------------------------------------------------------------");

            // Sort cards by their total purchases for better readability
            allGameCards.stream()
                .forEach(card -> {
                    CardStats stats = cardStatsMap.getOrDefault(card.name, new CardStats());
                    double winRate = stats.totalTimesPurchased > 0 ? (double) stats.timesPurchasedByWinner / stats.totalTimesPurchased : 0.0;
                    String conclusion = stats.totalTimesPurchased > 0 ? getBalanceConclusion(winRate) : "NO DATA";
                    out.printf("%-26s | %-15d | %-19d | %-24.2f%% | %s\n", 
                        card.name + " (" + card.cost + ")", 
                        stats.totalTimesPurchased, 
                        stats.timesPurchasedByWinner, 
                        winRate * 100,
                        conclusion
                    );
                });
            out.println("-----------------------------------------------------------------------------------");

        } catch (IOException e) {
            System.err.println("Could not write balance report: " + e.getMessage());
        }
    }

    /**
     * Simple logic to flag unbalanced cards based on average win rate (e.g., 50%).
     */
    private String getBalanceConclusion(double winRate) {
        // Average win rate is 50% for a balanced 2-player game.
        // We use a 5% buffer (45% to 55%) as the acceptable range.
        double threshold = 0.05; 
        double averageWinRate = 0.5;
        
        if (winRate > averageWinRate + threshold) {
            return "OVERPOWERED (Suggest Nerf)";
        } else if (winRate < averageWinRate - threshold) {
            return "UNDERPOWERED (Suggest Buff)";
        } else {
            return "BALANCED";
        }
    }

    public static void main(String[] args) {
        new CardBalanceSimulator().runSimulations();
    }
}
