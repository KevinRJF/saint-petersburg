import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;

import smile.classification.LogisticRegression;

public class MiMaSimulateGame {

	public static SPGameTranscript simulateGame(SPPlayer... players) {
		SPGameTranscript transcript = new SPGameTranscript();
		for (SPPlayer player : players) {
			transcript.addPlayerName(player.getName());
		}
		MiMaState state = new MiMaState(players.length);
		transcript.addState(state.clone());
		while (!state.isGameOver()) {
			int currentPlayerIndex = state.playerTurn;
			SPPlayer currentPlayer = players[currentPlayerIndex];
			MiMaState stateCopy = state.clone();
			int actionIndex = currentPlayer.getAction(stateCopy);
			ArrayList<SPAction> legalActions = stateCopy.getLegalActions();
			if (actionIndex < 0 || actionIndex >= legalActions.size()) {
				throw new IllegalArgumentException("Player " + currentPlayer.getName() + " chose an invalid action index: " + actionIndex);
			}
			SPAction action = legalActions.get(actionIndex);
			transcript.addAction(action);
			action.take();
			state = stateCopy;
			transcript.addState(state);
		}
		
		return transcript;
	}		

	public static void printGameTranscript(SPGameTranscript transcript, PrintStream out) {

		// Print the states and actions of the game:
		ArrayList<SPAction> actions = transcript.getActions();
		int numActions = actions.size();
		ArrayList<SPState> states = transcript.getStates();
		for (int i = 0; i < numActions; i++) {
			out.println(states.get(i));
			out.println("> " + actions.get(i) + "\n");
		}
		// Print the final state:
		out.println(states.get(states.size() - 1));
		boolean[] isWinner = transcript.getWinners();
		ArrayList<String> playerNames = transcript.getPlayerNames();
		for (int i = 0; i < isWinner.length; i++) {
			if (isWinner[i]) {
				out.printf("Player %d (%s) wins!", (i + 1), playerNames.get(i));
			}
		}
	}

	public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException {
		
//		java.io.ObjectInputStream ois = new java.io.ObjectInputStream(new java.io.FileInputStream("MiMaLogistic18.model"));
//        LogisticRegression.Binomial model = (LogisticRegression.Binomial) ois.readObject();
//        
//		
//		MiMaStateFeatures stateFeatures = new MiMaStateFeatures();
//		ArrayList<SPFeature> features = stateFeatures.features;
//		
//		System.out.println("Model coefficients:");
//        System.out.println(features.size() + " features");
//        System.out.println(model.coefficients().length + " coefficients");
//        System.out.printf("%.4f\tIntercept%n", model.coefficients()[0]);
//        for (int i = 0; i < model.coefficients().length - 1; i++) {
//            System.out.printf("%.4f\t%s%n", model.coefficients()[i + 1], features.get(i).getName());
//        }
//        
//        ois.close();
		
		SPGameTranscript transcript = simulateGame(new MiMaPlayer(), new SPPlayerFlatMC());
		
		
		boolean toFile = true; // Change to true to write to a file
		if (toFile) {
			try (PrintStream out = new PrintStream("game_transcript.txt")) {
				printGameTranscript(transcript, out);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		else {
			printGameTranscript(transcript, System.out);
		}
	}
	
}
