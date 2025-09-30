import java.util.ArrayList;

/**
 * Custom player implementation: SPMMustafizPlayer
 * This player evaluates possible actions based on defined features
 * and chooses the one with the highest score.
 */
public class MaiNePlayer extends SPPlayer {

   int numSimulationsPerAction = 10000;
   int playoutTerminationDepth = 4;
   MaiNeLR2 features = new MaiNeLR2();
   boolean verbose = false;

   public MaiNePlayer() {
      super("SPMMustafizPlayer");
   }

   public int getAction(SPState state) {
      int bestActionIndex = -1;
      double bestValue = Double.NEGATIVE_INFINITY;
      ArrayList<SPAction> actions = state.getLegalActions();
      int numActions = actions.size();
      if (this.verbose) {
         System.out.println("Number of legal actions: " + numActions);
      }

      for(int i = 0; i < numActions; ++i) {
         SPState depth1Copy = state.clone();
         SPAction action = (SPAction)depth1Copy.getLegalActions().get(i);
         action.take();
         double estValue = 0.0;

         for(int j = 0; j < this.numSimulationsPerAction; ++j) {
            SPState simCopy = depth1Copy.clone();

            for(int k = 0; !simCopy.isGameOver() && k < this.playoutTerminationDepth; ++k) {
               ArrayList<SPAction> legalActions = simCopy.getLegalActions();
               SPAction randomAction = (SPAction)legalActions.get((int)(Math.random() * (double)legalActions.size()));
               randomAction.take();
            }

            double heuristicValue = this.eval(simCopy);
            if (state.playerTurn != simCopy.playerTurn) {
               heuristicValue = 1.0 - heuristicValue;
            }

            estValue += heuristicValue;
         }

         estValue /= (double)this.numSimulationsPerAction;
         if (estValue > bestValue) {
            bestValue = estValue;
            bestActionIndex = i;
         }
      }

      if (this.verbose) {
         System.out.printf("%s (est. value %.4f)\n", actions.get(bestActionIndex), bestValue);
      }

      return bestActionIndex;
   }
   
   /**
	 * Checks if the given card is a unique aristocrat card for a specific player.
	 * A card is unique if it is an aristocrat and the player does not already
	 * own that aristocrat type.
	 *
	 * @param player the player index (0-based)
	 * @param card   the card to check
	 * @return true if the card is a unique aristocrat for that player, false otherwise
	 */

	/// Check with Neller
		public boolean isUniqueAristocratCard(SPState state, SPCard card) {
			// TODO Auto-generated method stub
			return state.playerAristocrats.get(state.playerTurn).stream().distinct() != null;
		}
   
   private double roundAwareHeuristicEval(SPState state) {
	    int round = state.round;
	    boolean isEndGame = state.isGameOver();
	    int player = state.playerTurn;

	    double totalScore = 0.0;
	    int cardCount = 0;

	    ArrayList<SPCard> allOwnedCards = new ArrayList<>();
	    allOwnedCards.addAll(state.playerWorkers.get(player));
	    allOwnedCards.addAll(state.playerBuildings.get(player));
	    allOwnedCards.addAll(state.playerAristocrats.get(player));

	    for (SPCard card : allOwnedCards) {
	        String type = card.type.toLowerCase();
	        double cardScore = 0.0;

	        // Always value workers, less in end game
	        if (type.equals("worker")) {
	            cardScore = 1.0 / (1 + card.cost);
	            if (isEndGame) cardScore *= 0.8;
	        }

	        // Mid game heuristic
	        if (!isEndGame && round > 3) {
	            if (type.equals("aristocrat") || type.equals("trading")) {
	                cardScore = card.rubles * 0.4 + card.points;
	            }
	        }

	        // End game: prioritize aristocrats, trading, points
	        if (isEndGame) {
	            if (type.equals("aristocrat") && isUniqueAristocratCard(state, card)) {
	                cardScore = 3.0 + card.points;
	            } else if (type.equals("aristocrat") || type.equals("trading")) {
	                cardScore = card.points + card.rubles * 0.5;
	            } else {
	                cardScore = card.points;
	            }
	        }

	        // Early game
	        if (round <= 3 && (type.equals("worker") || type.equals("building"))) {
	            cardScore = 1.0 / (1 + card.cost);
	        }

	        totalScore += cardScore;
	        cardCount++;
	    }

	    if (cardCount > 0) {
	        totalScore /= cardCount; // normalize
	    }

	    return totalScore;
	}


   private double eval(SPState state) {
	    double modelPrediction = this.features.predict(state); // logistic regression model
	    double heuristicScore = roundAwareHeuristicEval(state); // new helper method below

	    double alpha = 0.95;  // weight for ML model, tune as needed
	    return alpha * modelPrediction + (1 - alpha) * heuristicScore;
	}
}