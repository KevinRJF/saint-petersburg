import java.util.ArrayList;
import java.util.Random;

public class OKTurnBasedFeaturesPlayer extends SPPlayer {

    private final int numSimulationsPerAction = 1000;
    private final int playoutTerminationDepth = 15;
    private final boolean verbose = true;
    OKStateFeaturesLR1 features = new OKStateFeaturesLR1();
    private final Random rng = new Random();
    private final SPFeature roi = features.new ROIFeature();

    public OKTurnBasedFeaturesPlayer() {
        super("OKTurnBasedFeaturesPlayer");
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
            System.out.printf("ROIPlayer: %s (est. value %.4f)\n", actions.get(bestActionIndex), bestValue);
      }
        return bestActionIndex;
   }

    private double eval(SPState state) {
        return features.predict(state);
    }
}
