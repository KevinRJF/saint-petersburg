import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import java.util.Collections;
import java.io.*;

public class AiDanExpectiminimaxPlayer extends SPPlayer {

    int searchDepth = 8; 
    SPStateFeaturesLR1 features = new SPStateFeaturesLR1();
    boolean verbose = false;
    boolean useTranspositionTable = true;
    int chanceSamples = 3; // Number of random outcomes to sample at chance nodes
    Random rng = new Random();

    public AiDanExpectiminimaxPlayer() {
        super("AiDanExpectiminimaxPlayer");
    }

    public void setSearchDepth(int depth) {
        this.searchDepth = depth;
    }

    public void setUseTranspositionTable(boolean use) {
        this.useTranspositionTable = use;
    }
    
    public void setChanceSamples(int samples) {
        this.chanceSamples = samples;
    }

    @Override
    public int getAction(SPState state) {
        ArrayList<SPAction> actions = state.getLegalActions();
        double bestValue = Double.NEGATIVE_INFINITY;
        int bestActionIndex = -1;
        int originalPlayer = state.playerTurn;

        if (verbose) System.out.println("Number of legal actions: " + actions.size());

        double alpha = Double.NEGATIVE_INFINITY;
        double beta = Double.POSITIVE_INFINITY;
        
        // Root: always maximizing player's turn
        for (int i = 0; i < actions.size(); i++) {
            SPState copy = state.clone();
            ArrayList<SPAction> clonedActions = copy.getLegalActions();
            SPAction actionToTake = clonedActions.get(i);
            
            actionToTake.take(); 
            
            // Alpha-beta with chance node sampling
            double value = alphaBeta(copy, searchDepth - 1, originalPlayer, alpha, beta);

            if (verbose) System.out.printf("Action %s evaluated to %.4f\n", actions.get(i), value);

            if (value > bestValue) {
                bestValue = value;
                bestActionIndex = i;
            }
            
            alpha = Math.max(alpha, bestValue);
        }

        if (bestActionIndex == -1) {
             return 0; 
        }

        if (verbose) {
            System.out.printf("Chosen Action: %s (Value %.4f)\n", actions.get(bestActionIndex), bestValue);
            System.out.printf("TT Stats: Size=%d, Hits=%d, Stores=%d\n", 
                transTable.size(), ttHits, ttStores);
        }
        
        return bestActionIndex;
    }

    /**
     * Alpha-beta minimax with Monte Carlo sampling at chance nodes.
     * When random events occur (like Observatory draws), samples a few outcomes
     * and averages them instead of assuming a single deterministic outcome.
     */
    private double alphaBeta(SPState state, int depth, int originalPlayer, double alpha, double beta) {
        long hash = 0;
        
        // Compute Zobrist hash if using TT
        if (useTranspositionTable) {
            hash = AiDanZHashing.computeHash(state);
            
            // Check transposition table
            TTEntry entry = transTable.get(hash);
            if (entry != null && entry.depth >= depth) {
                ttHits++;
                
                // Exact value
                if (entry.lowerBound == entry.upperBound) {
                    return entry.lowerBound;
                }
                
                // Use bounds to prune
                if (entry.lowerBound >= beta) {
                    return entry.lowerBound;
                }
                if (entry.upperBound <= alpha) {
                    return entry.upperBound;
                }
                
                // Narrow the search window
                alpha = Math.max(alpha, entry.lowerBound);
                beta = Math.min(beta, entry.upperBound);
            }
        }
        
        // Base case
        if (depth == 0 || state.isGameOver()) {
            double rawValue = features.predict(state);
            double value = (state.playerTurn == originalPlayer || state.isGameOver()) 
                ? rawValue : 1.0 - rawValue;
            
            if (useTranspositionTable) {
                transTable.put(hash, new TTEntry(value, value, depth));
                ttStores++;
            }
            
            return value;
        }

        ArrayList<SPAction> actions = state.getLegalActions();
        if (actions.isEmpty()) {
            double value = features.predict(state);
            if (useTranspositionTable) {
                transTable.put(hash, new TTEntry(value, value, depth));
                ttStores++;
            }
            return value;
        }

        boolean maximizing = (state.playerTurn == originalPlayer);
        double origAlpha = alpha;
        double origBeta = beta;
        
        if (maximizing) {
            double maxEval = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < actions.size(); i++) {
                // Sample multiple random outcomes for this action
                double sampledValue = sampleAction(state, i, depth, originalPlayer, alpha, beta);
                
                maxEval = Math.max(maxEval, sampledValue);
                alpha = Math.max(alpha, maxEval);
                
                if (beta <= alpha) break;
            }
            
            if (useTranspositionTable) {
                double lower = maxEval;
                double upper = (maxEval >= origBeta) ? Double.POSITIVE_INFINITY : maxEval;
                
                if (maxEval <= origAlpha) {
                    upper = maxEval;
                    lower = Double.NEGATIVE_INFINITY;
                }
                
                transTable.put(hash, new TTEntry(lower, upper, depth));
                ttStores++;
            }
            
            return maxEval;
        } else {
            double minEval = Double.POSITIVE_INFINITY;
            for (int i = 0; i < actions.size(); i++) {
                // Sample multiple random outcomes for this action
                double sampledValue = sampleAction(state, i, depth, originalPlayer, alpha, beta);
                
                minEval = Math.min(minEval, sampledValue);
                beta = Math.min(beta, minEval);
                
                if (beta <= alpha) break;
            }
            
            if (useTranspositionTable) {
                double upper = minEval;
                double lower = (minEval <= origAlpha) ? Double.NEGATIVE_INFINITY : minEval;
                
                if (minEval >= origBeta) {
                    lower = minEval;
                    upper = Double.POSITIVE_INFINITY;
                }
                
                transTable.put(hash, new TTEntry(lower, upper, depth));
                ttStores++;
            }
            
            return minEval;
        }
    }
    
    /**
     * Sample an action multiple times with different random seeds to handle randomness.
     * Returns the average value across samples.
     */
    private double sampleAction(SPState state, int actionIndex, int depth, 
                                int originalPlayer, double alpha, double beta) {
        double totalValue = 0.0;
        
        for (int sample = 0; sample < chanceSamples; sample++) {
            SPState copy = state.clone();
            
            // Shuffle decks slightly to simulate randomness in draws
            if (sample > 0) { // First sample uses original ordering
                Collections.shuffle(copy.workerDeck, rng);
                Collections.shuffle(copy.buildingDeck, rng);
                Collections.shuffle(copy.aristocratDeck, rng);
                Collections.shuffle(copy.tradingDeck, rng);
            }
            
            ArrayList<SPAction> clonedActions = copy.getLegalActions();
            clonedActions.get(actionIndex).take();
            
            double value = alphaBeta(copy, depth - 1, originalPlayer, alpha, beta);
            totalValue += value;
        }
        
        return totalValue / chanceSamples;
    }

    public static class TTEntry {
        double lowerBound, upperBound;
        int depth;

        TTEntry(double lb, double ub, int d) { 
            lowerBound = lb; 
            upperBound = ub; 
            depth = d; 
        }
    }

    private Map<Long, TTEntry> transTable = new LinkedHashMap<>(1000000, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<Long, TTEntry> eldest) {
            return size() > 1000000;
        }
    };
    
    private long ttHits = 0;
    private long ttStores = 0;
    
    public void resetStats() {
        ttHits = 0;
        ttStores = 0;
    }
    
    public void clearTranspositionTable() {
        transTable.clear();
        resetStats();
    }
    
    public String getTTStats() {
        return String.format("TT: Size=%d, Hits=%d, Stores=%d, Hit Rate=%.2f%%", 
            transTable.size(), ttHits, ttStores, 
            ttStores > 0 ? (100.0 * ttHits / ttStores) : 0.0);
    }
    
}