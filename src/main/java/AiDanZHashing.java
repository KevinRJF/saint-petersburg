import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.HashMap;

//this class attempts to implement Zobrist hashing for SPState
public class AiDanZHashing {
    private static final int NUM_PLAYERS = 2;
    private static final int MAX_DECK_SIZE = 40; // Approx max deck size
    private static final int MAX_HOLDING_SIZE = 15; // Max workers/buildings/aristos per player
    private static final int MAX_HAND_SIZE = 5; // Max hand (3 base + warehouse)
    private static final int MAX_ROW_SIZE = 8; // Upper/lower rows
    private static final int MAX_RUBLES = 300; // Increased from 200 to handle higher accumulation
    private static final int MAX_POINTS = 300; // Max points
    private static final int MAX_ROUNDS = 15; // Increased for safety
    private static final int NUM_PHASES = 6; // Worker=0 to End=5 (includes Pub and End)
    private static final int NUM_DECK_TYPES = 4; // Worker=0, Building=1, Arist=2, Trading=3

    private static final long[] ZOBRIST_PHASE = new long[NUM_PHASES];
    private static final long[] ZOBRIST_PLAYER_TURN = new long[NUM_PLAYERS];
    private static final long[] ZOBRIST_ROUND = new long[MAX_ROUNDS];

    private static final long[] ZOBRIST_PLAYER_RUBLES = new long[NUM_PLAYERS * MAX_RUBLES];
    private static final long[] ZOBRIST_PLAYER_POINTS = new long[NUM_PLAYERS * MAX_POINTS];
    private static final long[] ZOBRIST_PLAYER_PASSED = new long[NUM_PLAYERS * 2]; // 0/1
    private static final long[] ZOBRIST_USED_OBS = new long[NUM_PLAYERS * (MAX_ROUNDS + 1)]; // Up to rounds+1

    // Starting player: phase x player
    private static final long[] ZOBRIST_STARTING_PLAYER = new long[NUM_DECK_TYPES * NUM_PLAYERS];

    // Holdings: player x position x card
    private static final long[][] ZOBRIST_HAND_POS = new long[NUM_PLAYERS][MAX_HAND_SIZE];
    private static final long[][] ZOBRIST_WORKER_POS = new long[NUM_PLAYERS][MAX_HOLDING_SIZE];
    private static final long[][] ZOBRIST_BUILDING_POS = new long[NUM_PLAYERS][MAX_HOLDING_SIZE];
    private static final long[][] ZOBRIST_ARISTO_POS = new long[NUM_PLAYERS][MAX_HOLDING_SIZE];

    // Rows: position
    private static final long[] ZOBRIST_UPPER_ROW_POS = new long[MAX_ROW_SIZE];
    private static final long[] ZOBRIST_LOWER_ROW_POS = new long[MAX_ROW_SIZE];

    // Decks: deck_type x position
    private static final long[][] ZOBRIST_DECK_POS = new long[NUM_DECK_TYPES][MAX_DECK_SIZE];

    // Discard pile: position (order may not matter, but include positional)
    private static final long[] ZOBRIST_DISCARD_POS = new long[MAX_DECK_SIZE];

    // Observed: null or card_id
    private static final long ZOBRIST_OBSERVED_NONE;
    // For observed card, use card key + fixed pos

    // Card keys: index by position in SPCard.ALL_CARDS (unique types)
    private static final long[] ZOBRIST_CARD;
    private static final Map<SPCard, Integer> CARD_INDEX; // Cache for quick lookup

    static {
        Random rand = new Random(42L); //uses same seed for reproducibility

        // Initialize phase, turn, round
        for (int i = 0; i < NUM_PHASES; i++) ZOBRIST_PHASE[i] = rand.nextLong();
        for (int i = 0; i < NUM_PLAYERS; i++) ZOBRIST_PLAYER_TURN[i] = rand.nextLong();
        for (int i = 0; i < MAX_ROUNDS; i++) ZOBRIST_ROUND[i] = rand.nextLong();

        // Player scalars
        for (long[] arr : new long[][]{ZOBRIST_PLAYER_RUBLES, ZOBRIST_PLAYER_POINTS}) {
            for (int i = 0; i < arr.length; i++) arr[i] = rand.nextLong();
        }
        for (int i = 0; i < NUM_PLAYERS; i++) {
            for (int j = 0; j < 2; j++) ZOBRIST_PLAYER_PASSED[i * 2 + j] = rand.nextLong();
        }
        for (int i = 0; i < NUM_PLAYERS * (MAX_ROUNDS + 1); i++) ZOBRIST_USED_OBS[i] = rand.nextLong();

        // Starting player
        for (int i = 0; i < NUM_DECK_TYPES * NUM_PLAYERS; i++) ZOBRIST_STARTING_PLAYER[i] = rand.nextLong();

        // Holdings pos
        for (long[][] arr2d : new long[][][]{ZOBRIST_HAND_POS, ZOBRIST_WORKER_POS, ZOBRIST_BUILDING_POS, ZOBRIST_ARISTO_POS}) {
            for (int p = 0; p < NUM_PLAYERS; p++) {
                for (int pos = 0; pos < arr2d[p].length; pos++) {
                    arr2d[p][pos] = rand.nextLong();
                }
            }
        }

        // Rows pos
        for (int i = 0; i < MAX_ROW_SIZE; i++) {
            ZOBRIST_UPPER_ROW_POS[i] = rand.nextLong();
            ZOBRIST_LOWER_ROW_POS[i] = rand.nextLong();
        }

        // Decks pos
        for (int d = 0; d < NUM_DECK_TYPES; d++) {
            for (int pos = 0; pos < MAX_DECK_SIZE; pos++) {
                ZOBRIST_DECK_POS[d][pos] = rand.nextLong();
            }
        }

        // Discard pos
        for (int i = 0; i < MAX_DECK_SIZE; i++) ZOBRIST_DISCARD_POS[i] = rand.nextLong();

        // Observed none
        ZOBRIST_OBSERVED_NONE = rand.nextLong();

        // Card keys: based on ALL_CARDS index
        List<SPCard> allCards = SPCard.ALL_CARDS;
        ZOBRIST_CARD = new long[allCards.size()];
        CARD_INDEX = new HashMap<>();
        for (int i = 0; i < allCards.size(); i++) {
            SPCard card = allCards.get(i);
            ZOBRIST_CARD[i] = rand.nextLong();
            CARD_INDEX.put(card, i); // Since cards are unique instances per type
        }
    }

    /**
     * Computes the Zobrist hash for the given SPState.
     * Assumes numPlayers=2; ignores isWinner as it's post-game.
     */
    public static long computeHash(SPState state) {
        long hash = 0;

        // Scalars
        hash ^= ZOBRIST_PHASE[state.phase];
        hash ^= ZOBRIST_PLAYER_TURN[state.playerTurn];
        hash ^= ZOBRIST_ROUND[(state.round - 1) % MAX_ROUNDS]; // 1-based to 0-index

        // Player scalars
        for (int p = 0; p < NUM_PLAYERS; p++) {
            hash ^= ZOBRIST_PLAYER_RUBLES[p * MAX_RUBLES + state.playerRubles[p]];
            hash ^= ZOBRIST_PLAYER_POINTS[p * MAX_POINTS + state.playerPoints[p]];
            long passedKey = state.playerPassed[p] ? ZOBRIST_PLAYER_PASSED[p * 2 + 1] : ZOBRIST_PLAYER_PASSED[p * 2 + 0];
            hash ^= passedKey;
            hash ^= ZOBRIST_USED_OBS[p * (MAX_ROUNDS + 1) + state.usedObservatories[p]];

            // Starting player per phase
            for (int ph = 0; ph < SPState.NUM_DECKS; ph++) {
                int sp = state.startingPlayer[ph];
                hash ^= ZOBRIST_STARTING_PLAYER[ph * NUM_PLAYERS + sp];
            }

            // Holdings (positional; assumes order as in lists)
            hash ^= xorList(state.playerHands.get(p), ZOBRIST_HAND_POS[p]);
            hash ^= xorList(state.playerWorkers.get(p), ZOBRIST_WORKER_POS[p]);
            hash ^= xorList(state.playerBuildings.get(p), ZOBRIST_BUILDING_POS[p]);
            hash ^= xorList(state.playerAristocrats.get(p), ZOBRIST_ARISTO_POS[p]);
        }

        // Global lists
        hash ^= xorList(state.workerDeck, ZOBRIST_DECK_POS[0]);
        hash ^= xorList(state.buildingDeck, ZOBRIST_DECK_POS[1]);
        hash ^= xorList(state.aristocratDeck, ZOBRIST_DECK_POS[2]);
        hash ^= xorList(state.tradingDeck, ZOBRIST_DECK_POS[3]);
        hash ^= xorList(state.upperCardRow, ZOBRIST_UPPER_ROW_POS);
        hash ^= xorList(state.lowerCardRow, ZOBRIST_LOWER_ROW_POS);
        hash ^= xorList(state.discardPile, ZOBRIST_DISCARD_POS);

        // Observed card
        if (state.observedCard != null) {
            Integer idx = CARD_INDEX.get(state.observedCard);
            if (idx != null) {
                hash ^= ZOBRIST_CARD[idx]; // + fixed pos if needed
            }
        } else {
            hash ^= ZOBRIST_OBSERVED_NONE;
        }

        return hash;
    }

    /**
     * XOR a list of cards: pos_key[i] ^ card_key for each position i.
     */
    private static long xorList(ArrayList<SPCard> list, long[] posKeys) {
        long h = 0;
        int size = Math.min(list.size(), posKeys.length);
        for (int i = 0; i < size; i++) {
            SPCard card = list.get(i);
            Integer idx = CARD_INDEX.get(card);
            if (idx != null) {
                h ^= posKeys[i] ^ ZOBRIST_CARD[idx];
            }
        }
        return h;
    }
}