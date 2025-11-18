import java.io.PrintStream;
import java.util.ArrayList;

public class SPSimulateGame {

	public static SPGameTranscript simulateGame(SPPlayer... players) {
		SPGameTranscript transcript = new SPGameTranscript();
		for (SPPlayer player : players) {
			transcript.addPlayerName(player.getName());
		}
		SPState state = new SPState(players.length);
		transcript.addState(state.clone());
		while (!state.isGameOver()) {
			int currentPlayerIndex = state.playerTurn;
			// System.out.println(currentPlayerIndex);
			SPPlayer currentPlayer = players[currentPlayerIndex];
			SPState stateCopy = state.clone();
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

	public static SPGameTranscript simulateGame(long millisPerGame, SPPlayer... players) {
		SPGameTranscript transcript = new SPGameTranscript();
		for (SPPlayer player : players) {
			transcript.addPlayerName(player.getName());
		}
		SPState state = new SPState(players.length);
		transcript.addState(state.clone());
		long[] millisRemaining = new long[] {millisPerGame/2, millisPerGame/2};
		while (!state.isGameOver()) {
			int currentPlayerIndex = state.playerTurn;
			// System.out.println(currentPlayerIndex);
			SPPlayer currentPlayer = players[currentPlayerIndex];
			SPState stateCopy = state.clone();
			long startTime = System.currentTimeMillis();
			int actionIndex = currentPlayer.getAction(stateCopy, 
					millisRemaining[currentPlayerIndex]);
			long elapsedTime = System.currentTimeMillis() - startTime;
			millisRemaining[currentPlayerIndex] -= elapsedTime;
			if (millisRemaining[currentPlayerIndex] < 0) {
				millisRemaining[currentPlayerIndex] = 0;
			}
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
				out.printf("Player %d (%s) wins!\n", (i + 1), playerNames.get(i));
			}
		}
	}

	public static void main(String[] args) {
		// Against random player using no hand:
		// SPGameTranscript transcript = simulateGame(new SPPlayerFlatMC(), new SPRandomNoHandPlayer());
		// SPGameTranscript transcript = simulateGame(new TWNPlayerFlatMCLinReg1(), new SPRandomNoHandPlayer());
		// SPGameTranscript transcript = simulateGame(new SPFlatMCNNPlayer(), new SPRandomNoHandPlayer());
		// SPGameTranscript transcript = simulateGame(new SPFlatMCNNPlayer(), new TWNFlatMCPlayer());

		// Against itself:
		// SPGameTranscript transcript = simulateGame(new SPPlayerFlatMC(), new SPPlayerFlatMC());
		
		// SPGameTranscript transcript = simulateGame(new SPMCTSPlayer(), new SPPlayerFlatMC());
		// SPGameTranscript transcript = simulateGame(new TWNMCTSPlayer2(), new SPPlayerFlatMC());
		//SPGameTranscript transcript = simulateGame(new TWNMCTSPlayer(), new TWNMCTSPlayer());
		SPGameTranscript transcript = simulateGame(60000, new TWNMCTSPlayer(), new TWNMCTSPlayer());

		boolean toFile = true; // Change to true to write to a file
		if (toFile) {
			try (PrintStream out = new PrintStream("game_transcript.txt")) {
				printGameTranscript(transcript, out);
			} catch (Exception e) {
				e.printStackTrace();
			}
			ArrayList<SPState> states = transcript.getStates();
			System.out.println(states.get(states.size() - 1));
		}
		else {
			printGameTranscript(transcript, System.out);
		}
	}
	
}
